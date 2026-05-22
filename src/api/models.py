"""
Models endpoints for fraud detection API.
"""
import os
import pickle
import numpy as np
from flask import Blueprint, request, jsonify

from .auth import require_api_key
from .db import save_transaction, save_prediction

models_bp = Blueprint('models', __name__, url_prefix='/api/v1')

# Model loading at startup
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/fraud_detection_model.pkl')
MODEL_THRESHOLD = float(os.getenv('MODEL_THRESHOLD', '0.4358'))
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')

_model = None
_feature_cols = None


def load_model():
    """Load the fraud detection model."""
    global _model, _feature_cols

    if _model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                artifacts = pickle.load(f)
                _model = artifacts['model']
                _feature_cols = artifacts['feature_cols']
                if 'threshold' in artifacts:
                    global MODEL_THRESHOLD
                    MODEL_THRESHOLD = artifacts['threshold']
            print(f"✓ Model loaded from {MODEL_PATH}")
            print(f"  Features: {len(_feature_cols)}")
            print(f"  Threshold: {MODEL_THRESHOLD}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise e

    return _model, _feature_cols


def prepare_features(data):
    """Prepare features for prediction."""
    model, feature_cols = load_model()

    # Get V features
    v_features = {}
    for i in range(1, 29):
        v_key = f'V{i}'
        v_features[v_key] = data.get(v_key, 0.0)

    # Calculate derived features
    amount = data.get('amount', 0.0)
    time_seconds = data.get('time_seconds', 0.0)

    amount_log = np.log1p(amount)
    time_hours = time_seconds / 3600 if time_seconds else 0
    is_high_amount = 1 if amount > 500 else 0
    is_very_high_amount = 1 if amount > 1000 else 0

    # Build feature vector in correct order
    feature_vector = []
    for col in feature_cols:
        if col.startswith('V'):
            feature_vector.append(v_features.get(col, 0.0))
        elif col == 'Amount_scaled':
            # Already scaled, use raw amount (will be scaled by model)
            feature_vector.append(amount)  # The model should scale this
        elif col == 'Amount_log':
            feature_vector.append(amount_log)
        elif col == 'Time_hours':
            feature_vector.append(time_hours)
        elif col == 'is_high_amount':
            feature_vector.append(is_high_amount)
        elif col == 'is_very_high_amount':
            feature_vector.append(is_very_high_amount)
        else:
            feature_vector.append(0.0)

    return np.array(feature_vector).reshape(1, -1)


@models_bp.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Predict fraud for a single transaction.

    Request body:
    {
        "transaction_id": "TX123",
        "time_seconds": 12345,
        "V1": -1.35, "V2": 0.27, ..., "V28": 0.02,
        "amount": 149.62
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Validate required fields
    required = ['V1', 'amount']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        # Prepare transaction data for DB
        transaction_data = {
            'transaction_id': data.get('transaction_id', f"TX_{data.get('time_seconds', 0)}"),
            'time_seconds': data.get('time_seconds', 0),
            'amount': data.get('amount', 0),
        }

        # Add all V features
        for i in range(1, 29):
            transaction_data[f'v{i}'] = data.get(f'V{i}', 0.0)

        # Save transaction to DB
        transaction_id = save_transaction(transaction_data)

        # Prepare features and predict
        X = prepare_features(data)
        model, _ = load_model()

        fraud_probability = float(model.predict_proba(X)[0, 1])
        predicted_class = 1 if fraud_probability >= MODEL_THRESHOLD else 0

        # Save prediction to DB
        save_prediction(transaction_id, predicted_class, fraud_probability, MODEL_VERSION)

        return jsonify({
            'transaction_id': transaction_id,
            'is_fraud': bool(predicted_class),
            'fraud_probability': round(fraud_probability, 4),
            'threshold_used': MODEL_THRESHOLD,
            'model_version': MODEL_VERSION
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@models_bp.route('/predict/batch', methods=['POST'])
@require_api_key
def predict_batch():
    """
    Predict fraud for multiple transactions.

    Request body:
    {
        "transactions": [
            {"transaction_id": "TX1", "V1": ..., "amount": ...},
            {"transaction_id": "TX2", "V1": ..., "amount": ...}
        ]
    }
    """
    data = request.get_json()

    if not data or 'transactions' not in data:
        return jsonify({'error': 'No transactions provided'}), 400

    transactions = data['transactions']

    if len(transactions) > 1000:
        return jsonify({'error': 'Maximum 1000 transactions per batch'}), 400

    results = []

    for tx in transactions:
        try:
            tx_data = {
                'transaction_id': tx.get('transaction_id', f"TX_{tx.get('time_seconds', 0)}"),
                'time_seconds': tx.get('time_seconds', 0),
                'amount': tx.get('amount', 0),
            }

            for i in range(1, 29):
                tx_data[f'v{i}'] = tx.get(f'V{i}', 0.0)

            transaction_id = save_transaction(tx_data)

            X = prepare_features(tx)
            model, _ = load_model()

            fraud_probability = float(model.predict_proba(X)[0, 1])
            predicted_class = 1 if fraud_probability >= MODEL_THRESHOLD else 0

            save_prediction(transaction_id, predicted_class, fraud_probability, MODEL_VERSION)

            results.append({
                'transaction_id': transaction_id,
                'is_fraud': bool(predicted_class),
                'fraud_probability': round(fraud_probability, 4)
            })

        except Exception as e:
            results.append({
                'transaction_id': tx.get('transaction_id', 'unknown'),
                'error': str(e)
            })

    fraud_count = sum(1 for r in results if r.get('is_fraud', False))

    return jsonify({
        'total_transactions': len(transactions),
        'frauds_detected': fraud_count,
        'results': results
    }), 200


@models_bp.route('/predictions', methods=['GET'])
@require_api_key
def get_predictions():
    """
    Get prediction history.

    Query params:
    - limit: Max records (default 100)
    - offset: Pagination offset (default 0)
    """
    from .db import get_prediction_history

    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    predictions = get_prediction_history(limit=limit, offset=offset)

    return jsonify({
        'count': len(predictions),
        'limit': limit,
        'offset': offset,
        'predictions': [
            {
                'id': p['id'],
                'transaction_id': p['transaction_id'],
                'is_fraud': bool(p['predicted_class']),
                'fraud_probability': round(p['fraud_probability'], 4),
                'amount': p['amount'],
                'model_version': p['model_version'],
                'prediction_time': str(p['prediction_time'])
            }
            for p in predictions
        ]
    }), 200


@models_bp.route('/predictions/<int:prediction_id>', methods=['GET'])
@require_api_key
def get_prediction(prediction_id):
    """Get a specific prediction by ID."""
    from .db import get_prediction_by_id

    prediction = get_prediction_by_id(prediction_id)

    if not prediction:
        return jsonify({'error': 'Prediction not found'}), 404

    return jsonify({
        'id': prediction['id'],
        'transaction_id': prediction['transaction_id'],
        'is_fraud': bool(prediction['predicted_class']),
        'fraud_probability': round(prediction['fraud_probability'], 4),
        'amount': prediction['amount'],
        'model_version': prediction['model_version'],
        'prediction_time': str(prediction['prediction_time']),
        'features': {
            f'V{i}': prediction[f'v{i}']
            for i in range(1, 29)
        }
    }), 200


@models_bp.route('/model/info', methods=['GET'])
@require_api_key
def model_info():
    """Get model information."""
    return jsonify({
        'model_version': MODEL_VERSION,
        'threshold': MODEL_THRESHOLD,
        'features_count': len(load_model()[1]) if _feature_cols else 0
    }), 200


@models_bp.route('/health', methods=['GET'])
def health():
    """Health check for the model service."""
    try:
        model, _ = load_model()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'threshold': MODEL_THRESHOLD
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
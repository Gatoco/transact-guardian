"""
Flask API for Payment Fraud Detection.

This API provides endpoints for fraud prediction and stores
transactions and predictions in PostgreSQL.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify

from .models import models_bp
from .db import check_db_health


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    app.config['JSON_SORT_KEYS'] = False

    # Register blueprints
    app.register_blueprint(models_bp)

    # Health check endpoint (no auth required)
    @app.route('/health', methods=['GET'])
    def health_check():
        """Main health check endpoint."""
        db_healthy = check_db_health()

        status = 'healthy' if db_healthy else 'degraded'
        http_status = 200 if db_healthy else 503

        return jsonify({
            'status': status,
            'service': 'fraud-detection-api',
            'version': os.getenv('MODEL_VERSION', '1.0.0'),
            'database': 'connected' if db_healthy else 'disconnected'
        }), http_status

    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API info."""
        return jsonify({
            'service': 'Payment Fraud Detection API',
            'version': '1.0.0',
            'endpoints': {
                'health': '/health',
                'predict': 'POST /api/v1/predict',
                'predict_batch': 'POST /api/v1/predict/batch',
                'predictions': 'GET /api/v1/predictions',
                'prediction_by_id': 'GET /api/v1/predictions/{id}',
                'model_info': 'GET /api/v1/model/info'
            },
            'authentication': 'Bearer API Key required for /api/* endpoints'
        }), 200

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500

    return app


# Create the app instance
app = create_app()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print(f"Starting Fraud Detection API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
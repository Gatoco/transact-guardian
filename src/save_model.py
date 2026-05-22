"""
Guardar y cargar el modelo entrenado.
"""
import pickle
import os

MODEL_PATH = 'models/fraud_detection_model.pkl'


def save_model(model, feature_cols, threshold=0.4358):
    """
    Guardar el modelo y sus configuraciones.

    Args:
        model: El modelo entrenado
        feature_cols: Lista de nombres de features
        threshold: Threshold óptimo encontrado
    """
    print("=" * 60)
    print("GUARDANDO MODELO")
    print("=" * 60)

    artifacts = {
        'model': model,
        'feature_cols': feature_cols,
        'threshold': threshold
    }

    os.makedirs('models', exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)

    print(f"✓ Modelo guardado en: {MODEL_PATH}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Threshold: {threshold:.4f}")


def load_model(path=MODEL_PATH):
    """Cargar el modelo guardado."""
    with open(path, 'rb') as f:
        artifacts = pickle.load(f)

    print(f"✓ Modelo cargado desde: {path}")
    print(f"  - Features: {len(artifacts['feature_cols'])}")
    print(f"  - Threshold: {artifacts['threshold']:.4f}")

    return artifacts['model'], artifacts['feature_cols'], artifacts['threshold']


def predict(model, X, threshold=0.5):
    """
    Hacer predicciones con el modelo.

    Args:
        model: Modelo cargado
        X: DataFrame o array con las features
        threshold: Threshold para clasificar

    Returns:
        y_pred: Predicciones (0 o 1)
        y_proba: Probabilidades de fraude
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred, y_proba


if __name__ == "__main__":
    # Test guardando y cargando
    from src.preprocess import preprocess_pipeline
    from sklearn.ensemble import RandomForestClassifier

    csv_path = 'creditcard.csv'
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Entrenar
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Guardar
    save_model(model, feature_cols, threshold=0.4358)

    # Cargar
    loaded_model, loaded_features, loaded_threshold = load_model()

    # Verificar que funciona
    y_pred, y_proba = predict(loaded_model, X_test[:5], threshold=loaded_threshold)
    print(f"\nTest predictions: {y_pred}")
    print(f"Probabilities: {y_proba}")
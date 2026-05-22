"""
Random Forest para Fraud Detection

Vamos a comparar Logistic Regression vs Random Forest.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def train_random_forest(X_train, y_train, class_weight='balanced', n_estimators=100):
    """
    Entrenar Random Forest.

    Parámetros importantes:
    - n_estimators: número de árboles (100 = bueno, 500 = mejor)
    - max_depth: profundidad máxima de cada árbol
    - class_weight: 'balanced' para handle imbalance
    """
    print("\n" + "=" * 60)
    print(f"ENTRENANDO RANDOM FOREST ({n_estimators} árboles)")
    print("=" * 60)

    print("""
Parámetros:
  n_estimators = 100    (100 árboles, bueno para empezar)
  max_depth = 10        (árboles no muy profundos)
  class_weight = 'balanced'
  n_jobs = -1           (usa todos los cores)
""")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1  # Usar todos los cores
    )

    print("Entrenando (puede tardar ~30 segundos)...")
    model.fit(X_train, y_train)
    print("✓ Entrenamiento completado")

    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluar modelo y mostrar métricas."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print(f"""
  Confusion Matrix:
                   Predicted
                Legit    Fraud
  Actual Legit  {tn:6,}  {fp:6,}
         Fraud   {fn:6,}  {tp:6,}

  Fraudes detectados: {tp}/75 ({tp/75*100:.1f}%)
  Fraudes escapados:  {fn}
  Falsas alarmas:     {fp}
""")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def get_feature_importance(model, feature_names):
    """Obtener las features más importantes."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n--- Top 10 Feature Importance ---")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

    return indices, importances


def compare_models(csv_path):
    """Comparar Logistic Regression vs Random Forest."""
    from src.preprocess import preprocess_pipeline
    from sklearn.linear_model import LogisticRegression

    print("=" * 60)
    print("COMPARACIÓN: LOGISTIC REGRESSION vs RANDOM FOREST")
    print("=" * 60)

    # Preprocess
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Logistic Regression
    print("\n" + "="*60)
    print("ENTRENANDO LOGISTIC REGRESSION")
    print("="*60)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Feature importance
    indices, importances = get_feature_importance(rf_model, feature_cols)

    # Comparación final
    print("\n" + "=" * 60)
    print("COMPARACIÓN FINAL")
    print("=" * 60)
    print(f"""
╔═══════════════════╦═══════════════════╦═══════════════════╗
║    Métrica         ║   Log. Regression  ║   Random Forest   ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Accuracy           ║  {lr_metrics['accuracy']:.4f}          ║  {rf_metrics['accuracy']:.4f}          ║
║ Precision          ║  {lr_metrics['precision']:.4f}          ║  {rf_metrics['precision']:.4f}          ║
║ Recall             ║  {lr_metrics['recall']:.4f}          ║  {rf_metrics['recall']:.4f}          ║
║ F1-Score           ║  {lr_metrics['f1']:.4f}          ║  {rf_metrics['f1']:.4f}          ║
║ ROC-AUC            ║  {lr_metrics['roc_auc']:.4f}          ║  {rf_metrics['roc_auc']:.4f}          ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Fraudes detectados ║  {lr_metrics['tp']:3d}/75          ║  {rf_metrics['tp']:3d}/75          ║
║ Fraudes escapados  ║  {lr_metrics['fn']:3d}             ║  {rf_metrics['fn']:3d}             ║
╚═══════════════════╩═══════════════════╩═══════════════════╝
""")

    if rf_metrics['recall'] > lr_metrics['recall']:
        print("✓ Random Forest detecta MÁS fraudes")
    if rf_metrics['roc_auc'] > lr_metrics['roc_auc']:
        print("✓ Random Forest tiene MEJOR ROC-AUC")

    return rf_model, rf_metrics


if __name__ == "__main__":
    csv_path = '/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv'
    model, metrics = compare_models(csv_path)
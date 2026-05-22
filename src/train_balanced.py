"""
Training module: Logistic Regression CON Class Weights

Aquí mostramos cómo class_weight='balanced' resuelve el imbalance.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def train_with_class_weights(X_train, y_train):
    """
    Entrenar modelo CON class_weight='balanced'.

    Esto le dice al modelo: "los fraudes son más importantes".
    """
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELO CON CLASS WEIGHTS")
    print("=" * 60)

    print("""
class_weight='balanced' hace lo siguiente:
- Calcula: peso_clase_0 = n_muestras / (2 * n_clase_0)
- Calcula: peso_clase_1 = n_muestras / (2 * n_clase_1)
- Aplica estos pesos en la función de pérdida
""")

    model = LogisticRegression(
        class_weight='balanced',  # ← LA CLAVE
        max_iter=1000,
        random_state=42
    )

    print("Entrenando...")
    model.fit(X_train, y_train)
    print("✓ Entrenamiento completado")

    return model


def compare_models(metrics_naive, metrics_balanced, y_test):
    """
    Comparar modelo naive vs balanced.
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN: NAIVE vs CLASS WEIGHTS")
    print("=" * 60)

    print(f"""
╔═══════════════════╦═══════════════════╦═══════════════════╗
║    Métrica         ║      Naive         ║    Balanced       ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Accuracy           ║  {metrics_naive['accuracy']:.4f}          ║  {metrics_balanced['accuracy']:.4f}          ║
║ Precision          ║  {metrics_naive['precision']:.4f}          ║  {metrics_balanced['precision']:.4f}          ║
║ Recall             ║  {metrics_naive['recall']:.4f}          ║  {metrics_balanced['recall']:.4f}          ║
║ F1-Score           ║  {metrics_naive['f1']:.4f}          ║  {metrics_balanced['f1']:.4f}          ║
║ ROC-AUC            ║  {metrics_naive['roc_auc']:.4f}          ║  {metrics_balanced['roc_auc']:.4f}          ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Fraudes detectados ║  {metrics_naive['tp']:3d}/{len(y_test[y_test==1])}           ║  {metrics_balanced['tp']:3d}/{len(y_test[y_test==1])}           ║
║ Fraudes escapados  ║  {metrics_naive['fn']:3d}             ║  {metrics_balanced['fn']:3d}             ║
║ Falsas alarmas     ║  {metrics_naive['fp']:5d}             ║  {metrics_balanced['fp']:5d}             ║
╚═══════════════════╩═══════════════════╩═══════════════════╝
""")

    improvement_recall = (metrics_balanced['recall'] - metrics_naive['recall']) / metrics_naive['recall'] * 100
    extra_detected = metrics_balanced['tp'] - metrics_naive['tp']

    print(f"""
ANÁLISIS:
─────────
✓ Recall mejoró: {metrics_naive['recall']:.2f} → {metrics_balanced['recall']:.2f} (+{improvement_recall:.1f}%)

""")

    if extra_detected > 0:
        print(f"✓ Detectamos {extra_detected} fraudes MÁS que antes")

    if metrics_balanced['fp'] > metrics_naive['fp']:
        extra_fp = metrics_balanced['fp'] - metrics_naive['fp']
        print(f"⚠️ Pero tenemos {extra_fp} falsas alarmas EXTRA")

    print(f"""
TRADE-OFF:
──────────
El modelo balanced detecta más fraudes (+{extra_detected if extra_detected > 0 else 0})
pero a costa de más falsas alarmas (+{metrics_balanced['fp'] - metrics_naive['fp'] if metrics_balanced['fp'] > metrics_naive['fp'] else 0})

Esto es un TRADE-OFF clásico en fraud detection:
- Recall ↑ (más fraudes detectados)
- Precision ↓ (más falsas alarmas)
""")


def full_pipeline(csv_path):
    """Pipeline completo: naive vs balanced."""
    from src.preprocess import preprocess_pipeline

    # Preprocess
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Importar métricas naive si existen
    from src.train_naive import train_naive_model, evaluate_naive_model
    model_naive, metrics_naive = train_naive_model(X_train, y_train), evaluate_naive_model(model_naive, X_test, y_test)

    # Train con class weights
    model_balanced = train_with_class_weights(X_train, y_train)

    # Evaluate balanced
    y_pred_balanced = model_balanced.predict(X_test)
    y_proba_balanced = model_balanced.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred_balanced)
    recall = recall_score(y_test, y_pred_balanced)
    f1 = f1_score(y_test, y_pred_balanced)
    roc_auc = roc_auc_score(y_test, y_proba_balanced)
    cm = confusion_matrix(y_test, y_pred_balanced)
    tn, fp, fn, tp = cm.ravel()

    metrics_balanced = {
        'accuracy': accuracy_score(y_test, y_pred_balanced),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

    print(f"\n--- Balanced Results ---")
    print(f"  Accuracy:  {metrics_balanced['accuracy']:.4f}")
    print(f"  Precision: {metrics_balanced['precision']:.4f}")
    print(f"  Recall:    {metrics_balanced['recall']:.4f}")
    print(f"  F1-Score:  {metrics_balanced['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics_balanced['roc_auc']:.4f}")

    print(f"""
--- Confusion Matrix (Balanced) ---
                 Predicted
              Legit    Fraud
Actual Legit  {tn:6,}  {fp:6,}
       Fraud   {fn:6,}  {tp:6,}
""")

    # Compare
    compare_models(metrics_naive, metrics_balanced, y_test)

    return model_balanced, metrics_balanced


if __name__ == "__main__":
    csv_path = '/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv'
    model, metrics = full_pipeline(csv_path)
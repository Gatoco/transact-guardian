"""
Training module: Baseline Naive (SIN class weight handling)

El objetivo de este script es MOSTRAR el problema del class imbalance.
Este modelo NO usa class weights para que veas por qué falla.
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


def train_naive_model(X_train, y_train):
    """
    Entrenar modelo SIN class weight (naive baseline).

    Esto es lo que haría alguien que no sabe del imbalance.
    """
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELO NAIVE (sin class weight)")
    print("=" * 60)

    model = LogisticRegression(max_iter=1000, random_state=42)

    print("\nEntrenando...")
    model.fit(X_train, y_train)
    print("✓ Entrenamiento completado")

    return model


def evaluate_naive_model(model, X_test, y_test, dataset_name="Test"):
    """
    Evaluar el modelo naive y mostrar TODAS las métricas.

    Acá vas a ver el problema: alta accuracy pero mal recall.
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN EN {dataset_name.upper()}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- Métricas básicas ---")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- Confusion Matrix ---")
    print(f"                 Predicted")
    print(f"              Legit    Fraud")
    print(f"Actual Legit  {tn:6,}  {fp:6,}")
    print(f"       Fraud   {fn:6,}  {tp:6,}")

    # Business impact
    print(f"\n--- Impacto Business ---")
    print(f"  Fraudes NO detectados (FN): {fn}")
    print(f"  Falsas alarmas (FP):         {fp}")
    print(f"  Fraudes detectados (TP):     {tp}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def explain_the_problem(metrics, y_test):
    """
    Explicar por qué este modelo falla.
    """
    print("\n" + "=" * 60)
    print("⚠️  EL PROBLEMA DEL IMBALANCE")
    print("=" * 60)

    fraud_rate = y_test.mean()
    predicted_fraud_rate = (metrics['tp'] + metrics['fp']) / len(y_test)

    print(f"""
Observa estos números:

1. Fraud rate real en test:    {fraud_rate:.4%} ({y_test.sum()} fraudes de {len(y_test):,})
2. Fraudes detectados (recall): {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)
   → Solo detectamos el {metrics['recall']*100:.1f}% de los fraudes reales!
   → {metrics['fn']} fraudes se nos ESCAPARON

3. Accuracy: {metrics['accuracy']:.4f} (parece bueno, pero...)

POR QUÉ SUCEDE ESTO:
────────────────────
El modelo aprendió que si siempre predice "Legit" (0),
acertará el {1-fraud_rate:.1f}% de las veces.

El modelo está "optimizado" para predecir la clase mayoritaria,
no para detectar fraudes.

RESULTADO:
──────────
✓ Accuracy alta ({metrics['accuracy']:.2f}%)
✗ Recall BAJO ({metrics['recall']:.2f}%)
✗ Muchos fraudes NO detectados ({metrics['fn']})

Esto es exactamente el problema del class imbalance.
""")

    print("\n" + "=" * 60)
    print("¿QUÉ HACEMOS PARA MEJORAR?")
    print("=" * 60)
    print("""
OPCIÓN 1: Class Weights
  → Dar más peso a la clase minoritaria (fraudes)
  → sklearn: class_weight='balanced'

OPCIÓN 2: Undersampling
  → Reducir legítimas para balancear

OPCIÓN 3: SMOTE (Oversampling)
  → Crear fraudes sintéticos

OPCIÓN 4: Ajustar Threshold
  → Bajar el threshold de 0.5 a ~0.1
""")

    return metrics


def full_pipeline(csv_path):
    """Pipeline completo para demostrar el problema."""
    from src.preprocess import preprocess_pipeline

    # Preprocess
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Train naive model
    model = train_naive_model(X_train, y_train)

    # Evaluate on test
    metrics = evaluate_naive_model(model, X_test, y_test)

    # Explain the problem
    explain_the_problem(metrics, y_test)

    return model, metrics


if __name__ == "__main__":
    csv_path = '/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv'
    model, metrics = full_pipeline(csv_path)
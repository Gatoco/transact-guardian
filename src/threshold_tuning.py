"""
Threshold Adjustment - Encontrar el balance óptimo

El threshold (umbral) es el valor de probabilidad a partir del cual
el modelo dice "esto es fraude".

Por defecto es 0.5, pero podemos ajustarlo.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def find_optimal_threshold(y_true, y_proba):
    """
    Encontrar el threshold que maximiza F1-Score.

    F1 es el balance entre Precision y Recall.
    """
    print("\n" + "=" * 60)
    print("BUSCANDO THRESHOLD ÓPTIMO")
    print("=" * 60)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calcular F1 para cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # El último threshold es 1.0, lo ignoramos
    f1_scores = f1_scores[:-1]
    thresholds = thresholds[:-1]

    # Encontrar el mejor
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print(f"""
┌────────────────────────────────────────────────────────┐
│  THRESHOLD ÓPTIMO (maximiza F1): {best_threshold:.4f}             │
├────────────────────────────────────────────────────────┤
│  F1-Score:     {best_f1:.4f}                                 │
│  Precision:    {best_precision:.4f}                                 │
│  Recall:       {best_recall:.4f}                                 │
└────────────────────────────────────────────────────────┘
""")

    return best_threshold, best_f1, precisions, recalls, thresholds, f1_scores


def compare_thresholds(y_true, y_proba, thresholds_to_test):
    """
    Comparar diferentes thresholds.
    """
    print("\n--- Comparación de Thresholds ---")
    print(f"{'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)

    results = []

    for thresh in thresholds_to_test:
        y_pred = (y_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = precision_score(y_true, y_pred) if tp + fp > 0 else 0
        recall = recall_score(y_true, y_pred) if tp + fn > 0 else 0
        f1 = f1_score(y_true, y_pred) if precision + recall > 0 else 0

        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })

        print(f"{thresh:<12.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {tp:>6} {fp:>6} {fn:>6}")

    return results


def plot_precision_recall_curve(thresholds, precisions, recalls, best_threshold, best_idx):
    """Crear plot de Precision-Recall vs Threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision y Recall vs Threshold
    ax1.plot(thresholds, precisions[:-1], 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    ax1.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.3f}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision y Recall vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: F1 vs Threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    ax2.plot(thresholds, f1_scores, 'purple', linewidth=2)
    ax2.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.3f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/03_threshold_optimization.png', dpi=100)
    print("✓ Guardado: notebooks/03_threshold_optimization.png")
    plt.close()


def full_pipeline(csv_path):
    """Pipeline completo de threshold optimization."""
    from src.preprocess import preprocess_pipeline
    from sklearn.linear_model import LogisticRegression

    # Preprocess
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Train con class weights
    print("\nEntrenando Logistic Regression con class_weight='balanced'...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    best_threshold, best_f1, precisions, recalls, thresholds, f1_scores = find_optimal_threshold(y_test, y_proba)

    # Compare different thresholds
    results = compare_thresholds(
        y_test, y_proba,
        thresholds_to_test=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Plot
    plot_precision_recall_curve(thresholds, precisions, recalls, best_threshold, np.argmax(f1_scores))

    # Final evaluation with optimal threshold
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()

    print(f"""
╔════════════════════════════════════════════════════════╗
║  RESULTADO CON THRESHOLD ÓPTIMO ({best_threshold:.3f})                  ║
╠════════════════════════════════════════════════════════╣
║  Confusion Matrix:                                     ║
║                 Predicted                              ║
║              Legit    Fraud                            ║
║  Actual Legit  {tn:6,}  {fp:6,}                            ║
║         Fraud   {fn:6,}  {tp:6,}                            ║
╠════════════════════════════════════════════════════════╣
║  Fraudes detectados: {tp}/75 ({tp/75*100:.1f}%)                            ║
║  Fraudes escapados:  {fn}                                       ║
║  Falsas alarmas:    {fp}                                      ║
╚════════════════════════════════════════════════════════╝
""")

    return best_threshold, results


if __name__ == "__main__":
    csv_path = '/home/iwakura/Documentos/github-projects/prachub/build-a-payment-fraud-detection-model/creditcard.csv'
    best_threshold, results = full_pipeline(csv_path)
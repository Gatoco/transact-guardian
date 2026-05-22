"""
Evaluation module for fraud detection models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(y_true, y_pred, y_proba, threshold=0.5):
    """
    Comprehensive model evaluation for fraud detection.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        dict: Dictionary of metrics
    """
    y_pred_binary = (y_proba >= threshold).astype(int)

    metrics = {
        'confusion_matrix': confusion_matrix(y_true, y_pred_binary).tolist(),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'auprc': average_precision_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'threshold_used': threshold,
        'fraud_rate_predicted': y_pred_binary.mean(),
        'fraud_rate_actual': y_true.mean()
    }

    return metrics


def find_optimal_threshold(y_true, y_proba, target_recall=0.80):
    """
    Find threshold that achieves target recall while maximizing precision.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_recall: Desired recall level

    Returns:
        dict: Threshold and corresponding metrics
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    valid_indices = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_indices) == 0:
        best_idx = 0
    else:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]

    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (y_proba >= best_threshold).astype(int)

    return {
        'threshold': float(best_threshold),
        'recall': float(recalls[best_idx]),
        'precision': float(precisions[best_idx]),
        'f1_score': float(f1_score(y_true, y_pred))
    }


def print_evaluation_report(metrics, dataset_name='Validation'):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT - {dataset_name}")
    print(f"{'='*60}")

    print(f"\nThreshold: {metrics['threshold_used']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    print(f"\nActual Fraud Rate: {metrics['fraud_rate_actual']:.4%}")
    print(f"Predicted Fraud Rate: {metrics['fraud_rate_predicted']:.4%}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Legit    Fraud")
    print(f"Actual Legit    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Fraud     {cm[1][0]:6d}  {cm[1][1]:6d}")


def create_metrics_dataframe(results_dict):
    """Create comparison DataFrame from multiple model results."""
    records = []

    for model_name, metrics in results_dict.items():
        record = {
            'model': model_name,
            'roc_auc': metrics['roc_auc'],
            'auprc': metrics['auprc'],
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
        records.append(record)

    return pd.DataFrame(records).sort_values('roc_auc', ascending=False)


def business_impact_analysis(metrics, cost_fn=10, cost_fp=1, cost_fn_mult=100):
    """
    Estimate business impact of the model.

    Args:
        metrics: Model metrics dictionary
        cost_fn: Cost per false negative (missed fraud)
        cost_fp: Cost per false positive (review effort)
        cost_fn_mult: Multiplier for fraud cost (fraud is more expensive)

    Returns:
        dict: Business impact analysis
    """
    cm = metrics['confusion_matrix']

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    total_cost = (fp * cost_fp) + (fn * cost_fn * cost_fn_mult)

    savings_with_model = (fn * cost_fn * cost_fn_mult) + (fp * cost_fp)

    baseline_cost = (tn + fp + fn + tp) * cost_fn * cost_fn_mult * metrics['fraud_rate_actual']

    return {
        'total_cost': total_cost,
        'baseline_cost': baseline_cost,
        'cost_reduction': baseline_cost - total_cost,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn
    }
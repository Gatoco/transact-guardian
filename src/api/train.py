#!/usr/bin/env python3
"""
Training script with MLflow tracking.

This script trains the fraud detection model and logs
all experiments to MLflow.
"""
import os
import sys
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)

from src.preprocess import preprocess_pipeline


def train_and_log_experiment(X_train, y_train, X_test, y_test, model_type='rf'):
    """
    Train a model and log results to MLflow.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: 'rf' for RandomForest, 'lr' for LogisticRegression

    Returns:
        model, metrics dict
    """
    with mlflow.start_run(run_name=f"{model_type}_balanced") as run:
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            mlflow.log_param('model_type', 'RandomForest')
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('max_depth', 10)
        else:
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train)
            mlflow.log_param('model_type', 'LogisticRegression')

        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = f1_scores[:-1].argmax()
        optimal_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'optimal_threshold': optimal_threshold,
            'best_f1': best_f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name="fraud_detection_model")

        return model, metrics, optimal_threshold


def main():
    """Main training pipeline with MLflow tracking."""
    print("=" * 60)
    print("TRAINING PIPELINE WITH MLFLOW TRACKING")
    print("=" * 60)

    # Configure MLflow
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    else:
        print("MLflow tracking URI not set, using local storage")

    # Set experiment
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud_detection')
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'creditcard.csv')

    # Skip if not exists (we'll use pre-trained model)
    if not os.path.exists(csv_path):
        print("Dataset not found. Using pre-trained model.")
        return

    print("\nPreprocessing data...")
    X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(csv_path)

    # Train Random Forest
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    rf_model, rf_metrics, rf_threshold = train_and_log_experiment(
        X_train, y_train, X_test, y_test, model_type='rf'
    )

    print(f"""
Random Forest Results:
  Accuracy:   {rf_metrics['accuracy']:.4f}
  Precision:  {rf_metrics['precision']:.4f}
  Recall:     {rf_metrics['recall']:.4f}
  F1-Score:   {rf_metrics['f1_score']:.4f}
  ROC-AUC:    {rf_metrics['roc_auc']:.4f}
  Threshold:  {rf_threshold:.4f}

  Fraudes detectados: {rf_metrics['tp']}/{(rf_metrics['tp']+rf_metrics['fn'])}
  Falsas alarmas:     {rf_metrics['fp']}
""")

    # Train Logistic Regression for comparison
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION (baseline)")
    print("=" * 60)

    lr_model, lr_metrics, lr_threshold = train_and_log_experiment(
        X_train, y_train, X_test, y_test, model_type='lr'
    )

    print(f"""
Logistic Regression Results:
  Accuracy:   {lr_metrics['accuracy']:.4f}
  Precision:  {lr_metrics['precision']:.4f}
  Recall:     {lr_metrics['recall']:.4f}
  F1-Score:   {lr_metrics['f1_score']:.4f}
  ROC-AUC:    {lr_metrics['roc_auc']:.4f}
  Threshold:  {lr_threshold:.4f}
""")

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nView experiments at: http://localhost:5001")
    print(f"(MLflow UI)")


if __name__ == "__main__":
    main()
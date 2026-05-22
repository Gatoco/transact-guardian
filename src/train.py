"""
Training module for fraud detection models.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import lightgbm as lgb
import pickle


def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """Train Logistic Regression baseline."""
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, class_weight='balanced', n_estimators=100):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, is_unbalance=True):
    """Train LightGBM classifier."""
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        is_unbalance=is_unbalance,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def find_optimal_threshold(y_true, y_proba):
    """Find optimal threshold to maximize F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def evaluate_model(model, X, y_true, threshold=0.5):
    """Evaluate model and return metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred,
                                                        target_names=['Legit', 'Fraud'],
                                                        output_dict=True),
        'predictions': y_pred,
        'probabilities': y_proba
    }

    return metrics


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train multiple models and compare performance."""
    results = {}

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_val, y_val)
    results['LogisticRegression'] = {'model': lr_model, 'metrics': lr_metrics}
    print(f"  ROC-AUC: {lr_metrics['roc_auc']:.4f}, F1: {lr_metrics['f1_score']:.4f}")

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_val, y_val)
    results['RandomForest'] = {'model': rf_model, 'metrics': rf_metrics}
    print(f"  ROC-AUC: {rf_metrics['roc_auc']:.4f}, F1: {rf_metrics['f1_score']:.4f}")

    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_metrics = evaluate_model(lgb_model, X_val, y_val)
    results['LightGBM'] = {'model': lgb_model, 'metrics': lgb_metrics}
    print(f"  ROC-AUC: {lgb_metrics['roc_auc']:.4f}, F1: {lgb_metrics['f1_score']:.4f}")

    return results


def get_best_model(results):
    """Get best model based on ROC-AUC."""
    best_name = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
    return best_name, results[best_name]['model']


def save_model(model, path):
    """Save model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path):
    """Load model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
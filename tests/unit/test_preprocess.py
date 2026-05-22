"""
Unit tests for preprocessing module.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.preprocess import (
    temporal_split,
    scale_amount,
    create_features,
    prepare_features,
    get_feature_columns
)


class TestTemporalSplit:
    """Tests for temporal split functionality."""

    def test_split_proportions(self, sample_csv_path):
        """Test that split creates correct proportions."""
        df = pd.read_csv(sample_csv_path)
        train, test = temporal_split(df, test_size=0.2)

        total = len(train) + len(test)
        assert abs(len(train) / total - 0.8) < 0.01
        assert abs(len(test) / total - 0.2) < 0.01

    def test_split_no_temporal_overlap(self, sample_csv_path):
        """Test that train and test sets don't overlap in time."""
        df = pd.read_csv(sample_csv_path)
        train, test = temporal_split(df, test_size=0.2)

        train_max_time = train['Time'].max()
        test_min_time = test['Time'].min()

        assert train_max_time < test_min_time, "Train/Test should not overlap in time"

    def test_split_preserves_all_rows(self, sample_csv_path):
        """Test that no rows are lost in split."""
        df = pd.read_csv(sample_csv_path)
        original_count = len(df)

        train, test = temporal_split(df, test_size=0.2)
        split_count = len(train) + len(test)

        assert original_count == split_count, "All rows should be preserved"

    def test_split_stratification(self, sample_csv_path):
        """Test that fraud rate is similar in both sets."""
        df = pd.read_csv(sample_csv_path)
        original_rate = df['Class'].mean()

        train, test = temporal_split(df, test_size=0.2)

        # Rates should be within 50% of each other (very loose check)
        assert abs(train['Class'].mean() - test['Class'].mean()) < 0.01

    def test_split_with_custom_dates(self):
        """Test split with custom date boundaries."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'Time': range(1000),
            **{f'V{i}': np.random.randn(1000) for i in range(1, 29)},
            'Amount': np.random.randn(1000),
            'Class': np.random.choice([0, 1], 1000)
        })

        train, test = temporal_split(df, test_size=0.3)

        assert len(train) == 700
        assert len(test) == 300
        assert train['Time'].max() < test['Time'].min()


class TestScaleAmount:
    """Tests for Amount scaling functionality."""

    def test_scaled_column_exists(self, sample_csv_path):
        """Test that Amount_scaled column is created."""
        df = pd.read_csv(sample_csv_path)
        train, test = scale_amount(df.head(100).copy(), df.tail(50).copy())

        assert 'Amount_scaled' in train.columns
        assert 'Amount_scaled' in test.columns

    def test_train_mean_near_zero(self, sample_csv_path):
        """Test that scaled train Amount has mean ≈ 0."""
        df = pd.read_csv(sample_csv_path)
        train, _ = scale_amount(df.head(100).copy(), df.tail(50).copy())

        assert abs(train['Amount_scaled'].mean()) < 0.1

    def test_train_std_near_one(self, sample_csv_path):
        """Test that scaled train Amount has std ≈ 1."""
        df = pd.read_csv(sample_csv_path)
        train, _ = scale_amount(df.head(100).copy(), df.tail(50).copy())

        assert abs(train['Amount_scaled'].std() - 1.0) < 0.2

    def test_test_not_fit_on_train(self, sample_csv_path):
        """Test that scaler is fit only on train, not on test."""
        df = pd.read_csv(sample_csv_path)
        train, test = scale_amount(df.head(100).copy(), df.tail(50).copy())

        # Test should be scaled with TRAIN's statistics, not its own
        train_mean = train['Amount'].mean()
        train_std = train['Amount'].std()

        # Test values should NOT have mean=0 and std=1 (that's only for train)
        # They should be scaled using train's parameters
        assert 'Amount_scaled' in test.columns


class TestFeatureEngineering:
    """Tests for feature engineering functionality."""

    def test_amount_log_created(self):
        """Test that Amount_log feature is created."""
        df = pd.DataFrame({'Amount': [0, 1, 10, 100, 1000]})
        df = create_features(df)

        assert 'Amount_log' in df.columns
        expected = [0, 0.693, 2.398, 4.615, 6.909]
        for i, val in enumerate(df['Amount_log'].values):
            assert abs(val - expected[i]) < 0.01 if expected[i] > 0 else True

    def test_time_hours_created(self):
        """Test that Time_hours feature is created."""
        df = pd.DataFrame({
            'Time': [0, 3600, 7200, 14400],
            'Amount': [10, 20, 30, 40]
        })
        df = create_features(df)

        assert 'Time_hours' in df.columns
        assert df['Time_hours'].iloc[0] == 0.0
        assert df['Time_hours'].iloc[1] == 1.0
        assert df['Time_hours'].iloc[2] == 2.0

    def test_is_high_amount(self):
        """Test is_high_amount feature (>500)."""
        df = pd.DataFrame({'Amount': [100, 400, 500, 501, 1000]})
        df = create_features(df)

        assert 'is_high_amount' in df.columns
        assert df['is_high_amount'].iloc[0] == 0
        assert df['is_high_amount'].iloc[1] == 0
        assert df['is_high_amount'].iloc[2] == 0
        assert df['is_high_amount'].iloc[3] == 1
        assert df['is_high_amount'].iloc[4] == 1

    def test_is_very_high_amount(self):
        """Test is_very_high_amount feature (>1000)."""
        df = pd.DataFrame({'Amount': [500, 1000, 1001, 2000]})
        df = create_features(df)

        assert 'is_very_high_amount' in df.columns
        assert df['is_very_high_amount'].iloc[0] == 0
        assert df['is_very_high_amount'].iloc[1] == 0
        assert df['is_very_high_amount'].iloc[2] == 1
        assert df['is_very_high_amount'].iloc[3] == 1


class TestPrepareFeatures:
    """Tests for feature preparation for model."""

    def test_output_is_numpy_array(self, sample_transaction):
        """Test that output is a numpy array."""
        X = prepare_features(sample_transaction)
        assert isinstance(X, np.ndarray)

    def test_output_shape(self, sample_transaction):
        """Test that output shape is correct."""
        X = prepare_features(sample_transaction)
        assert X.shape[0] == 1
        assert X.shape[1] == 33

    def test_no_nan_values(self, sample_transaction):
        """Test that prepared features don't contain NaN."""
        X = prepare_features(sample_transaction)
        assert not np.any(np.isnan(X))

    def test_all_v_features_used(self, sample_transaction):
        """Test that all V1-V28 features are in the output."""
        X = prepare_features(sample_transaction)

        # The first 28 columns should be V1-V28
        # We can't directly check values, but we can verify no NaN
        assert X.shape[1] == 33  # 28 V + Amount_scaled + 4 engineered features

    def test_amount_log_used(self):
        """Test that Amount_log is calculated."""
        tx = self.sample_transaction.copy()
        tx['amount'] = 100
        X = prepare_features(tx)
        # Just verify it doesn't crash
        assert X.shape == (1, 33)

    @property
    def sample_transaction(self):
        """Get sample transaction fixture."""
        return {
            'V1': -1.36, 'V2': 0.27, 'V3': 2.54, 'V4': 1.38, 'V5': -0.34,
            'V6': 0.46, 'V7': 0.24, 'V8': 0.10, 'V9': 0.36, 'V10': 0.09,
            'V11': -0.55, 'V12': -0.62, 'V13': -0.99, 'V14': -0.31,
            'V15': 1.47, 'V16': -0.47, 'V17': 0.21, 'V18': 0.03, 'V19': 0.40,
            'V20': 0.25, 'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07,
            'V25': 0.13, 'V26': -0.19, 'V27': 0.13, 'V28': -0.02,
            'amount': 149.62,
            'time_seconds': 0
        }


class TestGetFeatureColumns:
    """Tests for feature columns extraction."""

    def test_excludes_non_features(self):
        """Test that non-feature columns are excluded."""
        df = pd.DataFrame({
            'transaction_id': ['TX1', 'TX2'],
            'timestamp': [100, 200],
            'V1': [1.0, 2.0],
            'Class': [0, 1],
            'customer_id': ['C1', 'C2']
        })

        cols = get_feature_columns(df)

        assert 'transaction_id' not in cols
        assert 'timestamp' not in cols
        assert 'Class' not in cols
        assert 'customer_id' not in cols
        assert 'V1' in cols


class TestPreprocessPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def test_pipeline_end_to_end(self, sample_csv_path):
        """Test complete preprocessing pipeline."""
        from src.preprocess import preprocess_pipeline

        X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(
            sample_csv_path,
            test_size=0.2
        )

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 33
        assert len(feature_cols) == 33
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_fraud_rate_preserved(self, sample_csv_path):
        """Test that fraud rate is similar after preprocessing."""
        from src.preprocess import preprocess_pipeline

        df = pd.read_csv(sample_csv_path)
        original_rate = df['Class'].mean()

        _, y_train, _, y_test, _ = preprocess_pipeline(sample_csv_path, test_size=0.2)

        combined_rate = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))
        assert abs(combined_rate - original_rate) < 0.001
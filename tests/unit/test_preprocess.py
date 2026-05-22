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
    load_data,
    temporal_split,
    scale_amount,
    create_features,
    preprocess_pipeline
)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dataframe(self, sample_csv_path):
        """Test that load_data returns a DataFrame."""
        df = load_data(sample_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_expected_columns(self, sample_csv_path):
        """Test that loaded data has expected columns."""
        df = load_data(sample_csv_path)
        assert 'Time' in df.columns
        assert 'V1' in df.columns
        assert 'Amount' in df.columns
        assert 'Class' in df.columns


class TestTemporalSplit:
    """Tests for temporal split functionality."""

    def test_split_proportions(self, sample_csv_path):
        """Test that split creates correct proportions."""
        df = load_data(sample_csv_path)
        train, test = temporal_split(df, test_size=0.2)

        total = len(train) + len(test)
        assert abs(len(train) / total - 0.8) < 0.01
        assert abs(len(test) / total - 0.2) < 0.01

    def test_split_preserves_all_rows(self, sample_csv_path):
        """Test that no rows are lost in split."""
        df = load_data(sample_csv_path)
        original_count = len(df)

        train, test = temporal_split(df, test_size=0.2)
        split_count = len(train) + len(test)

        assert original_count == split_count

    def test_split_with_explicit_dataframe(self):
        """Test split with custom DataFrame."""
        df = pd.DataFrame({
            'Time': range(1000),
            **{f'V{i}': np.random.randn(1000) for i in range(1, 29)},
            'Amount': np.random.randn(1000),
            'Class': np.random.choice([0, 1], 1000)
        })

        train, test = temporal_split(df, test_size=0.3)

        assert len(train) == 700
        assert len(test) == 300


class TestScaleAmount:
    """Tests for Amount scaling functionality."""

    def test_scaled_column_exists(self, sample_csv_path):
        """Test that Amount_scaled column is created."""
        df = load_data(sample_csv_path)
        train, test = scale_amount(df.iloc[:100].copy(), df.iloc[100:150].copy())

        assert 'Amount_scaled' in train.columns
        assert 'Amount_scaled' in test.columns

    def test_train_mean_near_zero(self, sample_csv_path):
        """Test that scaled train Amount has mean ≈ 0."""
        df = load_data(sample_csv_path)
        train, _ = scale_amount(df.iloc[:100].copy(), df.iloc[100:150].copy())

        assert abs(train['Amount_scaled'].mean()) < 0.1

    def test_train_std_near_one(self, sample_csv_path):
        """Test that scaled train Amount has std ≈ 1."""
        df = load_data(sample_csv_path)
        train, _ = scale_amount(df.iloc[:100].copy(), df.iloc[100:150].copy())

        assert abs(train['Amount_scaled'].std() - 1.0) < 0.2


class TestFeatureEngineering:
    """Tests for feature engineering functionality."""

    def test_amount_log_created(self):
        """Test that Amount_log feature is created."""
        df = pd.DataFrame({
            'Time': [0, 100, 200],
            'Amount': [0, 1, 10]
        })
        df = create_features(df)

        assert 'Amount_log' in df.columns
        assert df['Amount_log'].iloc[0] == pytest.approx(0, abs=0.01)
        assert df['Amount_log'].iloc[1] == pytest.approx(0.693, abs=0.01)

    def test_time_hours_created(self):
        """Test that Time_hours feature is created."""
        df = pd.DataFrame({
            'Time': [0, 3600, 7200],
            'Amount': [10, 20, 30]
        })
        df = create_features(df)

        assert 'Time_hours' in df.columns
        assert df['Time_hours'].iloc[0] == 0.0
        assert df['Time_hours'].iloc[1] == 1.0

    def test_is_high_amount(self):
        """Test is_high_amount feature (>500)."""
        df = pd.DataFrame({
            'Time': [0, 1, 2, 3, 4],
            'Amount': [100, 400, 500, 501, 1000]
        })
        df = create_features(df)

        assert 'is_high_amount' in df.columns
        assert df['is_high_amount'].iloc[0] == 0
        assert df['is_high_amount'].iloc[1] == 0
        assert df['is_high_amount'].iloc[2] == 0
        assert df['is_high_amount'].iloc[3] == 1
        assert df['is_high_amount'].iloc[4] == 1

    def test_is_very_high_amount(self):
        """Test is_very_high_amount feature (>1000)."""
        df = pd.DataFrame({
            'Time': [0, 1, 2, 3],
            'Amount': [500, 1000, 1001, 2000]
        })
        df = create_features(df)

        assert 'is_very_high_amount' in df.columns
        assert df['is_very_high_amount'].iloc[0] == 0
        assert df['is_very_high_amount'].iloc[1] == 0
        assert df['is_very_high_amount'].iloc[2] == 1


class TestPreprocessPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def test_pipeline_end_to_end(self, sample_csv_path):
        """Test complete preprocessing pipeline."""
        X_train, y_train, X_test, y_test, feature_cols = preprocess_pipeline(
            sample_csv_path
        )

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 33
        assert len(feature_cols) == 33
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_fraud_rate_preserved(self, sample_csv_path):
        """Test that fraud rate is similar after preprocessing."""
        df = load_data(sample_csv_path)
        original_rate = df['Class'].mean()

        _, y_train, _, y_test, _ = preprocess_pipeline(sample_csv_path)

        combined_rate = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))
        assert abs(combined_rate - original_rate) < 0.001

    def test_feature_count_correct(self, sample_csv_path):
        """Test that correct number of features are generated."""
        _, _, _, _, feature_cols = preprocess_pipeline(sample_csv_path)

        assert len(feature_cols) == 33
        assert 'V1' in feature_cols
        assert 'V28' in feature_cols
        assert 'Amount_scaled' in feature_cols

    def test_no_nan_in_features(self, sample_csv_path):
        """Test that no NaN values in features."""
        X_train, _, _, _, _ = preprocess_pipeline(sample_csv_path)

        nan_count = np.isnan(X_train).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in features"
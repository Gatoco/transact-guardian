import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_sample_csv(path, n_samples=1000):
    """Helper to create sample dataset for testing."""
    np.random.seed(42)
    data = {
        'Time': np.random.uniform(0, 172792, n_samples),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Amount': np.random.lognormal(4, 1.5, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }
    pd.DataFrame(data).to_csv(path, index=False)


class TestPreprocessing:
    """Tests for preprocessing pipeline."""

    def test_temporal_split_no_overlap(self, tmp_path):
        """Test that temporal split doesn't overlap."""
        from src.preprocess import temporal_split

        csv_path = tmp_path / "test.csv"
        n = 1000
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
            'Time': range(n),
            **{f'V{i}': np.random.randn(n) for i in range(1, 29)},
            'Amount': np.random.randn(n),
            'Class': np.random.choice([0, 1], n)
        })
        df.to_csv(csv_path, index=False)

        train, test = temporal_split(pd.read_csv(csv_path), test_size=0.2)

        train_max_time = train['Time'].max()
        test_min_time = test['Time'].min()

        assert train_max_time < test_min_time, "Train/Test should not overlap in time"


    def test_scale_amount(self, tmp_path):
        """Test that Amount is scaled correctly."""
        from src.preprocess import scale_amount

        train = pd.DataFrame({'Amount': [10, 20, 30, 40, 50]})
        test = pd.DataFrame({'Amount': [15, 25, 35]})

        train_out, test_out = scale_amount(train.copy(), test.copy())

        assert 'Amount_scaled' in train_out.columns
        assert 'Amount_scaled' in test_out.columns

        assert abs(train_out['Amount_scaled'].mean()) < 0.1
        assert abs(train_out['Amount_scaled'].std() - 1.0) < 0.1


    def test_feature_engineering(self, tmp_path):
        """Test feature engineering creates correct features."""
        from src.preprocess import create_features

        df = pd.DataFrame({
            'Amount': [10, 100, 500, 1000, 2000],
            'Time': [0, 3600, 7200, 10800, 14400]
        })

        df = create_features(df)

        assert 'Amount_log' in df.columns
        assert 'Time_hours' in df.columns
        assert 'is_high_amount' in df.columns
        assert 'is_very_high_amount' in df.columns

        assert df['is_high_amount'].sum() == 3
        assert df['is_very_high_amount'].sum() == 1


class TestAuth:
    """Tests for authentication."""

    def test_validate_api_key(self):
        """Test API key validation."""
        from src.api.auth import validate_api_key, get_api_key

        os.environ['API_KEY'] = 'fk_test_key_123'

        assert validate_api_key('fk_test_key_123') == True
        assert validate_api_key('Bearer fk_test_key_123') == True
        assert validate_api_key('wrong_key') == False
        assert validate_api_key('') == False


class TestModels:
    """Tests for model endpoints."""

    def test_prepare_features_shape(self):
        """Test feature preparation returns correct shape."""
        from src.api.models import prepare_features

        data = {f'V{i}': 0.0 for i in range(1, 29)}
        data['amount'] = 100.0
        data['time_seconds'] = 3600.0

        X = prepare_features(data)

        assert X.shape[0] == 1
        assert X.shape[1] == 33


    def test_prepare_features_no_error(self):
        """Test feature preparation doesn't crash."""
        from src.api.models import prepare_features

        data = {f'V{i}': np.random.randn() for i in range(1, 29)}
        data['amount'] = 150.0
        data['time_seconds'] = 1000.0

        X = prepare_features(data)

        assert not np.any(np.isnan(X))


class TestDatabase:
    """Tests for database operations."""

    def test_db_connection_no_crash(self):
        """Test DB connection doesn't crash (even if DB is down)."""
        from src.api.db import check_db_health

        result = check_db_health()
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
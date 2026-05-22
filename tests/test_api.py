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


@pytest.mark.skipif(os.getenv('SKIP_API_TESTS') == '1', reason="Flask/psycopg2 not installed")
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


@pytest.mark.skipif(os.getenv('SKIP_API_TESTS') == '1', reason="Flask/psycopg2 not installed")
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


@pytest.mark.skipif(os.getenv('SKIP_API_TESTS') == '1', reason="Flask/psycopg2 not installed")
class TestDatabase:
    """Tests for database operations."""

    def test_db_connection_no_crash(self):
        """Test DB connection doesn't crash (even if DB is down)."""
        from src.api.db import check_db_health

        result = check_db_health()
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
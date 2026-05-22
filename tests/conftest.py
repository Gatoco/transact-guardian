"""
Global fixtures for all tests.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope='session')
def project_root():
    """Get project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='session')
def sample_csv_path(project_root, tmp_path_factory):
    """Create a sample CSV for testing if real dataset doesn't exist."""
    csv_path = os.path.join(project_root, 'creditcard.csv')
    if os.path.exists(csv_path):
        return csv_path
    # Create minimal test dataset
    tmp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = tmp_dir / "test.csv"
    create_sample_dataset(csv_path, n_samples=2000)
    return str(csv_path)


def create_sample_dataset(path, n_samples=1000, fraud_rate=0.01, seed=42):
    """Helper to create sample dataset for testing."""
    np.random.seed(seed)
    data = {
        'Time': np.random.uniform(0, 172792, n_samples),
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Amount': np.random.lognormal(4, 1.5, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture(scope='session')
def api_key():
    """API key for testing."""
    test_key = 'fk_test_key_12345'
    os.environ['API_KEY'] = test_key
    return test_key


@pytest.fixture
def app():
    """Create Flask app for testing."""
    os.environ['DATABASE_URL'] = 'postgresql://changeme:changeme@localhost:5432/fraud_detection'
    os.environ['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detection_model.pkl')
    os.environ['MODEL_THRESHOLD'] = '0.4358'
    os.environ['MODEL_VERSION'] = '1.0.0'

    from src.api.app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_transaction():
    """Sample transaction data for testing."""
    return {
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311170, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'amount': 149.62,
        'time_seconds': 0
    }


@pytest.fixture
def sample_transaction_fraud():
    """Sample fraud transaction for testing."""
    return {
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311170, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'amount': 2500.00,
        'time_seconds': 86400
    }


@pytest.fixture
def auth_headers(api_key):
    """Authorization headers for API requests."""
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }


@pytest.fixture
def db_config():
    """Database configuration for testing."""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'dbname': 'fraud_detection',
        'user': 'changeme',
        'password': 'changeme'
    }


@pytest.fixture
def is_integration():
    """Check if we're running integration tests with real DB."""
    return os.getenv('RUN_INTEGRATION_TESTS', '0') == '1'
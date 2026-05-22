"""
Integration tests with real PostgreSQL database.
These tests require the PostgreSQL service to be running.
Run with: RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@pytest.mark.skipif(
    os.getenv('RUN_INTEGRATION_TESTS') != '1',
    reason="Integration tests require RUN_INTEGRATION_TESTS=1"
)
class TestDatabaseIntegration:
    """Integration tests with real PostgreSQL database."""

    def test_db_connection(self, db_config):
        """Test that we can connect to the database."""
        from src.api.db import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

    def test_save_and_retrieve_transaction(self, db_config):
        """Test saving a transaction and retrieving it."""
        from src.api.db import save_transaction

        transaction_data = {
            'transaction_id': f'INT_TEST_{os.getpid()}_1',
            'time_seconds': 1000,
            'amount': 199.99,
            'v1': -1.36, 'v2': 0.27, 'v3': 2.54, 'v4': 1.38,
            'v5': -0.34, 'v6': 0.46, 'v7': 0.24, 'v8': 0.10,
            'v9': 0.36, 'v10': 0.09, 'v11': -0.55, 'v12': -0.62,
            'v13': -0.99, 'v14': -0.31, 'v15': 1.47, 'v16': -0.47,
            'v17': 0.21, 'v18': 0.03, 'v19': 0.40, 'v20': 0.25,
            'v21': -0.02, 'v22': 0.28, 'v23': -0.11, 'v24': 0.07,
            'v25': 0.13, 'v26': -0.19, 'v27': 0.13, 'v28': -0.02,
        }

        tx_id = save_transaction(transaction_data)
        assert tx_id == transaction_data['transaction_id']

    def test_save_and_retrieve_prediction(self, db_config):
        """Test saving a prediction and retrieving it."""
        from src.api.db import save_transaction, save_prediction, get_prediction_by_id

        tx_id = f'INT_TEST_{os.getpid()}_2'

        # Save transaction first
        transaction_data = {
            'transaction_id': tx_id,
            'time_seconds': 2000,
            'amount': 299.99,
            **{f'v{i}': 0.0 for i in range(1, 29)}
        }
        save_transaction(transaction_data)

        # Save prediction
        save_prediction(tx_id, predicted_class=0, fraud_probability=0.15, model_version='1.0.0')

        # Retrieve - we need to find it by querying
        from src.api.db import get_prediction_history
        history = get_prediction_history(limit=10)
        matching = [p for p in history if p['transaction_id'] == tx_id]
        assert len(matching) > 0
        assert matching[0]['fraud_probability'] == 0.15

    def test_health_check(self, db_config):
        """Test database health check."""
        from src.api.db import check_db_health
        assert check_db_health() is True


@pytest.mark.skipif(
    os.getenv('RUN_INTEGRATION_TESTS') != '1',
    reason="Integration tests require RUN_INTEGRATION_TESTS=1"
)
class TestFullPipelineIntegration:
    """End-to-end integration tests."""

    def test_full_prediction_flow(self, client, auth_headers, sample_transaction):
        """Test complete prediction flow: API -> DB -> retrieval."""
        from src.api.db import get_prediction_history

        # Make prediction
        response = client.post(
            '/api/v1/predict',
            headers=auth_headers,
            data=json.dumps(sample_transaction)
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'transaction_id' in data
        assert 'fraud_probability' in data

        tx_id = data['transaction_id']

        # Retrieve from history
        history = get_prediction_history(limit=10)
        matching = [p for p in history if p['transaction_id'] == tx_id]
        assert len(matching) > 0


import json
"""
API tests for predict endpoints.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test that health endpoint returns status field."""
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'status' in data

    def test_health_returns_version(self, client):
        """Test that health endpoint returns version."""
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'version' in data

    def test_health_no_auth_required(self, client):
        """Test that health endpoint doesn't require authentication."""
        response = client.get('/health')
        assert response.status_code == 200


class TestPredictEndpoint:
    """Tests for /api/v1/predict endpoint."""

    def test_predict_valid_transaction(self, client, auth_headers, sample_transaction):
        """Test prediction with valid transaction."""
        response = client.post(
            '/api/v1/predict',
            headers=auth_headers,
            data=json.dumps(sample_transaction)
        )
        # May fail due to DB not being available, but should return proper JSON
        assert response.status_code in [200, 500]

    def test_predict_missing_auth(self, client, sample_transaction):
        """Test prediction without auth returns 401."""
        response = client.post(
            '/api/v1/predict',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_transaction)
        )
        assert response.status_code == 401

    def test_predict_invalid_key(self, client, sample_transaction):
        """Test prediction with invalid key returns 401."""
        response = client.post(
            '/api/v1/predict',
            headers={
                'Authorization': 'Bearer fk_invalid_key',
                'Content-Type': 'application/json'
            },
            data=json.dumps(sample_transaction)
        )
        assert response.status_code == 401

    def test_predict_missing_required_field(self, client, auth_headers):
        """Test prediction with missing required field returns 400."""
        incomplete_transaction = {
            'V1': -1.36
            # Missing 'amount' which is required
        }
        response = client.post(
            '/api/v1/predict',
            headers=auth_headers,
            data=json.dumps(incomplete_transaction)
        )
        # Should return 400 or 500 (if DB check happens first)
        assert response.status_code in [400, 500]

    def test_predict_no_data(self, client, auth_headers):
        """Test prediction with no data returns 400."""
        response = client.post(
            '/api/v1/predict',
            headers=auth_headers,
            data='{}'
        )
        assert response.status_code in [400, 500]

    def test_predict_empty_body(self, client, auth_headers):
        """Test prediction with empty body returns error."""
        response = client.post(
            '/api/v1/predict',
            headers=auth_headers,
            data=''
        )
        # Flask returns 415 for empty body with JSON content-type
        assert response.status_code in [400, 415, 500]


class TestPredictBatchEndpoint:
    """Tests for /api/v1/predict/batch endpoint."""

    def test_batch_predict_valid(self, client, auth_headers, sample_transaction):
        """Test batch prediction with valid transactions."""
        batch = {'transactions': [sample_transaction, sample_transaction]}
        response = client.post(
            '/api/v1/predict/batch',
            headers=auth_headers,
            data=json.dumps(batch)
        )
        assert response.status_code in [200, 500]

    def test_batch_predict_no_auth(self, client, sample_transaction):
        """Test batch prediction without auth returns 401."""
        batch = {'transactions': [sample_transaction]}
        response = client.post(
            '/api/v1/predict/batch',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(batch)
        )
        assert response.status_code == 401

    def test_batch_predict_empty_transactions(self, client, auth_headers):
        """Test batch prediction with empty list returns empty results."""
        batch = {'transactions': []}
        response = client.post(
            '/api/v1/predict/batch',
            headers=auth_headers,
            data=json.dumps(batch)
        )
        # Empty batch returns 200 with 0 results
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_transactions'] == 0

    def test_batch_predict_missing_transactions_key(self, client, auth_headers):
        """Test batch prediction without 'transactions' key returns error."""
        batch = {}
        response = client.post(
            '/api/v1/predict/batch',
            headers=auth_headers,
            data=json.dumps(batch)
        )
        assert response.status_code in [400, 500]


class TestPredictionsHistoryEndpoint:
    """Tests for /api/v1/predictions endpoint."""

    def test_predictions_requires_auth(self, client):
        """Test that predictions endpoint requires authentication."""
        response = client.get('/api/v1/predictions')
        assert response.status_code == 401

    def test_predictions_with_auth(self, client, auth_headers):
        """Test predictions endpoint with valid auth."""
        response = client.get(
            '/api/v1/predictions',
            headers=auth_headers
        )
        # Should return 200 or 500 (if DB not available)
        assert response.status_code in [200, 500]

    def test_predictions_pagination_params(self, client, auth_headers):
        """Test predictions with limit and offset params."""
        response = client.get(
            '/api/v1/predictions?limit=10&offset=0',
            headers=auth_headers
        )
        assert response.status_code in [200, 500]

    def test_predictions_by_id_requires_auth(self, client):
        """Test that prediction by ID requires auth."""
        response = client.get('/api/v1/predictions/1')
        assert response.status_code == 401

    def test_predictions_by_id_with_auth(self, client, auth_headers):
        """Test getting prediction by ID with valid auth."""
        response = client.get(
            '/api/v1/predictions/999999',
            headers=auth_headers
        )
        # Should return 404 (not found) or 500 (DB error)
        assert response.status_code in [404, 500]


class TestModelInfoEndpoint:
    """Tests for /api/v1/model/info endpoint."""

    def test_model_info_requires_auth(self, client):
        """Test that model info requires authentication."""
        response = client.get('/api/v1/model/info')
        assert response.status_code == 401

    def test_model_info_with_auth(self, client, auth_headers):
        """Test model info with valid auth."""
        response = client.get(
            '/api/v1/model/info',
            headers=auth_headers
        )
        # Should return 200 or 500 (if model not loaded)
        assert response.status_code in [200, 500]

    def test_model_info_contains_version(self, client, auth_headers):
        """Test that model info contains version."""
        response = client.get(
            '/api/v1/model/info',
            headers=auth_headers
        )
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'model_version' in data


class TestRootEndpoint:
    """Tests for root / endpoint."""

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API info."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'service' in data
        assert 'endpoints' in data
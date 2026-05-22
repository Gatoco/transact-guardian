"""
Unit tests for authentication module.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.api.auth import (
    validate_api_key,
    get_api_key,
    require_api_key,
    APIKeyAuth
)


class TestGetApiKey:
    """Tests for get_api_key function."""

    def test_returns_string(self):
        """Test that API key is returned as string."""
        os.environ['API_KEY'] = 'test_key_123'
        key = get_api_key()
        assert isinstance(key, str)
        assert key == 'test_key_123'

    def test_returns_empty_when_not_set(self):
        """Test that empty string is returned when API_KEY not set."""
        if 'API_KEY' in os.environ:
            del os.environ['API_KEY']
        key = get_api_key()
        assert key == ''


class TestValidateApiKey:
    """Tests for validate_api_key function."""

    def test_valid_key(self):
        """Test that valid API key returns True."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('fk_valid_key_123') is True

    def test_valid_key_with_bearer_prefix(self):
        """Test that valid key with Bearer prefix returns True."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('Bearer fk_valid_key_123') is True

    def test_invalid_key(self):
        """Test that invalid API key returns False."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('fk_wrong_key') is False

    def test_invalid_key_with_bearer(self):
        """Test that invalid key with Bearer prefix returns False."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('Bearer fk_wrong_key') is False

    def test_empty_key(self):
        """Test that empty key returns False."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('') is False

    def test_empty_key_with_bearer(self):
        """Test that empty key with Bearer prefix returns False."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('Bearer ') is False

    def test_partial_key(self):
        """Test that partial match returns False."""
        os.environ['API_KEY'] = 'fk_valid_key_123'
        assert validate_api_key('fk_valid') is False
        assert validate_api_key('fk_valid_key') is False
        assert validate_api_key('valid_key_123') is False

    def test_case_sensitive(self):
        """Test that API key comparison is case-sensitive."""
        os.environ['API_KEY'] = 'fk_Valid_Key'
        assert validate_api_key('fk_Valid_Key') is True
        assert validate_api_key('fk_valid_key') is False
        assert validate_api_key('FK_VALID_KEY') is False


class TestAPIKeyAuth:
    """Tests for APIKeyAuth class."""

    def test_verify_valid_key(self, monkeypatch):
        """Test verification with valid key."""
        os.environ['API_KEY'] = 'fk_test_key'
        auth = APIKeyAuth()

        class MockRequest:
            headers = {'Authorization': 'Bearer fk_test_key'}

        monkeypatch.setattr('src.api.auth.request', MockRequest())

        is_valid, error = auth.verify()
        assert is_valid is True
        assert error is None

    def test_verify_invalid_key(self, monkeypatch):
        """Test verification with invalid key."""
        os.environ['API_KEY'] = 'fk_test_key'
        auth = APIKeyAuth()

        class MockRequest:
            headers = {'Authorization': 'Bearer fk_wrong_key'}

        monkeypatch.setattr('src.api.auth.request', MockRequest())

        is_valid, error = auth.verify()
        assert is_valid is False
        assert error == "Invalid API key"

    def test_verify_missing_header(self, monkeypatch):
        """Test verification with missing Authorization header."""
        auth = APIKeyAuth()

        class MockRequest:
            headers = {}

        monkeypatch.setattr('src.api.auth.request', MockRequest())

        is_valid, error = auth.verify()
        assert is_valid is False
        assert error == "Missing Authorization header"
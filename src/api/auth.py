"""
API Key authentication middleware for Flask.
"""
import os
import secrets
import functools
from flask import request, jsonify, current_app

API_KEY_HEADER = 'Authorization'
API_KEY_PREFIX = 'Bearer '


def get_api_key() -> str:
    """Get API key from environment."""
    return os.getenv('API_KEY', '')


def validate_api_key(api_key: str) -> bool:
    """Validate the provided API key."""
    if not api_key:
        return False

    # Remove Bearer prefix if present
    if api_key.startswith(API_KEY_PREFIX):
        api_key = api_key[len(API_KEY_PREFIX):]

    # Compare with stored key
    stored_key = get_api_key()
    return secrets.compare_digest(api_key, stored_key)


def require_api_key(f):
    """
    Decorator to require API key authentication on endpoints.

    Usage:
        @app.route('/protected')
        @require_api_key
        def protected_endpoint():
            return "Secret data"
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get(API_KEY_HEADER)

        if not auth_header:
            current_app.logger.warning("Missing Authorization header")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Missing Authorization header'
            }), 401

        if not validate_api_key(auth_header):
            current_app.logger.warning("Invalid API key attempt")
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid API key'
            }), 401

        return f(*args, **kwargs)

    return decorated_function


class APIKeyAuth:
    """Context manager for API key validation in routes."""

    @staticmethod
    def verify():
        """Verify the API key from request headers."""
        auth_header = request.headers.get(API_KEY_HEADER)

        if not auth_header:
            return False, "Missing Authorization header"

        if not validate_api_key(auth_header):
            return False, "Invalid API key"

        return True, None
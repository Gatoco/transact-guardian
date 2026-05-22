"""
Database connection and operations for fraud detection API.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://changeme:changeme@postgres:5432/fraud_detection')


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        if conn:
            conn.close()


@contextmanager
def get_db_cursor(commit=True):
    """Context manager for database cursors with auto-commit."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()


def save_transaction(transaction_data: Dict[str, Any]) -> str:
    """
    Save a transaction to the database.

    Args:
        transaction_data: Dict with transaction fields

    Returns:
        transaction_id: The ID of the saved transaction
    """
    transaction_id = transaction_data.get('transaction_id', f"TX_{transaction_data['time_seconds']}")

    sql = """
    INSERT INTO transactions (
        transaction_id, time_seconds,
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
        v21, v22, v23, v24, v25, v26, v27, v28,
        amount
    ) VALUES (
        %(transaction_id)s, %(time_seconds)s,
        %(v1)s, %(v2)s, %(v3)s, %(v4)s, %(v5)s, %(v6)s, %(v7)s, %(v8)s, %(v9)s, %(v10)s,
        %(v11)s, %(v12)s, %(v13)s, %(v14)s, %(v15)s, %(v16)s, %(v17)s, %(v18)s, %(v19)s, %(v20)s,
        %(v21)s, %(v22)s, %(v23)s, %(v24)s, %(v25)s, %(v26)s, %(v27)s, %(v28)s,
        %(amount)s
    )
    ON CONFLICT (transaction_id) DO NOTHING
    RETURNING id
    """

    with get_db_cursor() as cursor:
        cursor.execute(sql, transaction_data)
        result = cursor.fetchone()
        return transaction_id


def save_prediction(transaction_id: str, predicted_class: int, fraud_probability: float, model_version: str):
    """
    Save a prediction to the database.

    Args:
        transaction_id: The transaction ID
        predicted_class: 0 (legit) or 1 (fraud)
        fraud_probability: Probability of fraud
        model_version: Version of the model used
    """
    sql = """
    INSERT INTO predictions (transaction_id, predicted_class, fraud_probability, model_version)
    VALUES (%(transaction_id)s, %(predicted_class)s, %(fraud_probability)s, %(model_version)s)
    """

    with get_db_cursor() as cursor:
        cursor.execute(sql, {
            'transaction_id': transaction_id,
            'predicted_class': predicted_class,
            'fraud_probability': fraud_probability,
            'model_version': model_version
        })


def get_prediction_history(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get prediction history.

    Args:
        limit: Maximum number of records
        offset: Offset for pagination

    Returns:
        List of predictions with transaction data
    """
    sql = """
    SELECT
        p.id,
        p.transaction_id,
        p.predicted_class,
        p.fraud_probability,
        p.model_version,
        p.created_at as prediction_time,
        t.amount,
        t.time_seconds
    FROM predictions p
    JOIN transactions t ON p.transaction_id = t.transaction_id
    ORDER BY p.created_at DESC
    LIMIT %(limit)s OFFSET %(offset)s
    """

    with get_db_cursor() as cursor:
        cursor.execute(sql, {'limit': limit, 'offset': offset})
        return cursor.fetchall()


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific prediction by ID.

    Args:
        prediction_id: The prediction ID

    Returns:
        Prediction dict or None
    """
    sql = """
    SELECT
        p.id,
        p.transaction_id,
        p.predicted_class,
        p.fraud_probability,
        p.model_version,
        p.created_at as prediction_time,
        t.amount,
        t.time_seconds,
        t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8, t.v9, t.v10,
        t.v11, t.v12, t.v13, t.v14, t.v15, t.v16, t.v17, t.v18, t.v19, t.v20,
        t.v21, t.v22, t.v23, t.v24, t.v25, t.v26, t.v27, t.v28
    FROM predictions p
    JOIN transactions t ON p.transaction_id = t.transaction_id
    WHERE p.id = %(id)s
    """

    with get_db_cursor() as cursor:
        cursor.execute(sql, {'id': prediction_id})
        return cursor.fetchone()


def check_db_health() -> bool:
    """Check if database is accessible."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception:
        return False
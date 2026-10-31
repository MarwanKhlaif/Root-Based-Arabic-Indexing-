"""
Database Connection Manager for Arabic IR PostgreSQL 16.
Provides standardized connection parameters, context managers, and health checks.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, Dict, Any
import psycopg
from psycopg import Connection, Cursor
from psycopg.rows import dict_row


def get_db_config() -> Dict[str, Any]:
    """
    Retrieve PostgreSQL connection parameters from environment with fallback defaults.
    """
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "arabic_ir"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", "postgres_password"),
        "connect_timeout": int(os.environ.get("PGCONNECT_TIMEOUT", "5")),
        "client_encoding": "UTF8",
    }


def get_connection(autocommit: bool = False, as_dict: bool = False) -> Connection:
    """
    Establish and return a new psycopg 3 connection to the arabic_ir database.
    """
    config = get_db_config()
    row_factory = dict_row if as_dict else None
    try:
        conn = psycopg.connect(**config, autocommit=autocommit, row_factory=row_factory)
        return conn
    except psycopg.OperationalError as exc:
        print(f"\n[ERROR] Could not connect to PostgreSQL database '{config['dbname']}' on {config['host']}:{config['port']}.")
        print("Ensure the Docker container 'arabic_ir_db' is running: `docker compose ps`")
        print(f"Details: {exc}\n", file=sys.stderr)
        raise


@contextmanager
def get_cursor(autocommit: bool = True, as_dict: bool = False) -> Generator[Cursor, None, None]:
    """
    Context manager yielding a database cursor with automatic cleanup.
    """
    conn = get_connection(autocommit=autocommit, as_dict=as_dict)
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()

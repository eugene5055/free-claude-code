"""Track token usage per request for session analytics."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    # Table to track token usage per request
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS request_usage (
            request_id TEXT PRIMARY KEY NOT NULL,
            session_id TEXT,
            provider_id TEXT NOT NULL,
            model TEXT NOT NULL,
            wire_api TEXT NOT NULL CHECK(wire_api IN ('messages', 'responses')),
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0.0,
            duration_ms REAL NOT NULL DEFAULT 0.0,
            status_code INTEGER,
            error_type TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS request_usage_session ON request_usage(session_id, created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS request_usage_provider ON request_usage(provider_id, created_at DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS request_usage_model ON request_usage(model, created_at DESC)"
    )
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("qrank.database.sql.execution")


@dataclass(slots=True)
class ExecutionResult:
    rows: tuple[tuple, ...] = ()
    error: str | None = None
    time_out: bool = False

    @property
    def valid(self) -> bool:
        return self.error is None and not self.time_out

    @property
    def empty(self) -> bool:
        return self.valid and not self.rows


def execute_sql(
    database_path: str | Path,
    sql: str,
    timeout: float = 30.0,
    dialect: str = "sqlite",
) -> ExecutionResult:
    """Execute one query against a read-only SQLite database."""

    if dialect != "sqlite":
        raise ValueError(f"Unsupported dialect: {dialect}")
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    result = ExecutionResult()
    connection: sqlite3.Connection | None = None
    connection_ready = threading.Event()

    def run() -> None:
        nonlocal connection
        try:
            uri = f"{path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                timeout=timeout,
            )
            connection.execute("PRAGMA query_only = ON")
            connection_ready.set()
            cursor = connection.execute(sql)
            result.rows = tuple(cursor.fetchall())
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                result.time_out = True
            else:
                result.error = str(exc)
        except Exception as exc:
            result.error = str(exc)
        finally:
            connection_ready.set()
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.debug("Error closing connection", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        result.time_out = True
        connection_ready.wait(timeout=1.0)
        if connection is not None:
            try:
                connection.interrupt()
            except Exception:
                logger.debug("Error interrupting connection", exc_info=True)
        thread.join(timeout=2.0)
        if thread.is_alive():
            logger.warning("execute_sql: worker thread still alive after interrupt")

    # Return a snapshot so a slow daemon thread cannot mutate the caller's value.
    return ExecutionResult(
        rows=result.rows,
        error=result.error,
        time_out=result.time_out,
    )


__all__ = ["ExecutionResult", "execute_sql"]

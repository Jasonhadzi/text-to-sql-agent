"""Load CSV into DuckDB and produce a structured schema summary."""

from __future__ import annotations

import duckdb

from src.models.schemas import SchemaColumn, SchemaSummary, TableSchema

# Columns known to contain PII in the retail dataset
_PII_COLUMNS = {"name", "email", "phone", "address"}

# Allowed tables for the SQL validator
ALLOWED_TABLES = {"retail_transactions", "retail_transactions_typed"}


def load_csv_to_duckdb(csv_path: str) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and load the CSV as a table.

    Creates:
      - ``retail_transactions`` — raw import via ``read_csv_auto``
      - ``retail_transactions_typed`` — view with parsed date/time columns
    """
    conn = duckdb.connect(database=":memory:")

    # Load raw table
    conn.execute(
        f"""
        CREATE TABLE retail_transactions AS
        SELECT * FROM read_csv_auto('{csv_path}', header=true, ignore_errors=true)
        """
    )

    # Create typed view with aliased date/time columns.
    # DuckDB read_csv_auto already parses Date as DATE and Time as TIME,
    # so we just alias them for a consistent contract.
    conn.execute(
        """
        CREATE VIEW retail_transactions_typed AS
        SELECT
            *,
            "Date" AS date_parsed,
            "Time" AS time_parsed
        FROM retail_transactions
        """
    )

    return conn


def get_schema_summary(conn: duckdb.DuckDBPyConnection) -> SchemaSummary:
    """Query DuckDB for column metadata and return a SchemaSummary."""
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'retail_transactions_typed' "
        "ORDER BY ordinal_position"
    ).fetchall()

    columns: list[SchemaColumn] = []
    for col_name, col_type in rows:
        columns.append(
            SchemaColumn(
                name=col_name,
                type=col_type,
                pii=col_name.lower() in _PII_COLUMNS,
            )
        )

    table = TableSchema(
        name="retail_transactions_typed",
        description="Retail transactions from CSV with parsed date/time columns",
        columns=columns,
        primary_key_candidates=["Transaction_ID"],
    )

    row_count = conn.execute(
        "SELECT COUNT(*) FROM retail_transactions_typed"
    ).fetchone()[0]

    return SchemaSummary(
        tables=[table],
        recommended_table="retail_transactions_typed",
        notes=[f"Total rows: {row_count}"],
    )

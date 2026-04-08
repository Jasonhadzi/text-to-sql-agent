# Text-to-SQL Agent

A multi-agent system that answers natural language questions about your data by generating, validating, and executing SQL queries — then producing a human-friendly analysis.

Built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), [sqlglot](https://github.com/tobymao/sqlglot), and either **[DuckDB](https://duckdb.org/)** (local CSV) or **Microsoft Fabric Warehouse** over ODBC (optional production path).

## How It Works

```
User Question
     │
     ▼
┌──────────────────────┐
│ Input Guardrail      │  ← Safety patterns
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Clarification Agent  │  ← Is the question clear?
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Query Router Agent   │  ← Picks datasource / tables (from config)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ NLQ Agent            │  ← One SQL candidate (DuckDB or T-SQL)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ SQL Validator        │  ← sqlglot: read-only, allowlist, row cap
│ (deterministic)      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ SQL Executor         │  ← DuckDB or Fabric (pyodbc)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ RAG Agent            │  ← Grounded answer from preview rows
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Output Guardrail     │  ← Quality / citation checks
└──────────────────────┘
```

Stages are logged to `outputs/runs/{run_id}/events.jsonl`.

## Prerequisites

- Python 3.11+
- An OpenAI API key (for the full agent pipeline)
- **Fabric only:** [ODBC Driver 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) and a **service principal** with access to the warehouse (see `src/connectors/fabric_connector.py` for env vars)

## Setup

1. **Clone and install:**

```bash
cd text-to-sql-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional — Microsoft Fabric / pyodbc
pip install -e ".[fabric]"
```

2. **Set your API key:**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Local data (DuckDB path only):**

Default CSV is `new_retail_data 1.csv` in the project root (often symlinked under `data/sources/`). When Fabric env is active, the pipeline uses the warehouse instead of loading this CSV.

## Configuration

| File | Purpose |
|------|---------|
| `src/config/datasource_config.json` | Datasources, allowed tables per backend (`duckdb` vs `fabric`), `default_datasource` |
| `src/config/schema_config.json` | Column descriptions, PII flags, relationships — merged at runtime with live introspection when available |
| `src/config/allowlist_config.json` | Fallback allowlist if datasource config lists no tables |

**Fabric:** set `FABRIC_CONNECTION_STRING` *or* `FABRIC_SERVER` + `FABRIC_DATABASE` + `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET`. Point `default_datasource` at your `type: "fabric"` entry and list real `tables` / `sql_schema`.

## Usage

```bash
source .venv/bin/activate

# Local DuckDB + CSV (Fabric env vars unset)
python -m src.app --question "What are the top 5 product categories by total revenue?"

python -m src.app --question "Show monthly sales trends" --source "path/to/data.csv"
```

With Fabric configured, the same command runs against the warehouse; SQL is validated as **T-SQL** and row caps use **TOP** / **FETCH** where applicable.

### Example Questions

```bash
python -m src.app -q "What are the top 5 product categories by total revenue?"
python -m src.app -q "What is the average order value by country?"
python -m src.app -q "Show me monthly revenue trends for 2023"
python -m src.app -q "Which shipping method has the highest average rating?"
python -m src.app -q "What is the revenue split between customer segments?"
```

## Output

Each run writes under `outputs/runs/{run_id}/`:

| File | Description |
|------|-------------|
| `events.jsonl` | Append-only log of pipeline stages |
| `schema.json` | Schema shown to agents (introspected + merged from `schema_config.json`) |
| `query.sql` | Validated SQL (after any row-cap injection) |
| `result.parquet` | Full result set (**DuckDB** path, when `COPY` succeeds) |
| `result.json` | Full result set (**Fabric** path, or DuckDB `COPY` fallback) |
| `result_preview.json` | First rows of the result |
| `final.json` / `final.md` | Structured + Markdown final response |

## Safety Guarantees

- **Read-only SQL:** AST validation blocks DML/DDL; Fabric connector also opens ODBC **read-only** and allows only `SELECT` / `WITH`
- **Single statement:** Semicolons inside the statement are rejected
- **Table allowlist:** Tables must appear under the active backend in `datasource_config.json` (Fabric vs local are scoped separately)
- **PII:** Sensitive columns are redacted in previews per schema / PII config
- **Row cap:** Missing limits get a safe default (`LIMIT` for DuckDB, **TOP** / **FETCH** for T-SQL)

## Running Tests

```bash
pytest tests/ -v
```

Use `PYTHONPATH=.` from the repo root if you run pytest without an editable install.

## Project Structure

```
src/
  app.py                 # CLI entry point
  orchestrator.py        # Pipeline
  agents/                # Clarification, query router, NLQ, RAG, …
  connectors/            # Azure OpenAI, Fabric (ODBC)
  tools/                 # Schema, validate, execute, log, redact
  config/                # datasource_config.json, schema_config.json, …
  models/schemas.py      # Pydantic I/O contracts
  prompts/               # Agent prompt templates (Markdown)
tests/
data/sources/            # CSV files (DuckDB)
outputs/runs/            # Per-run artifacts
```

## Architecture

The orchestrator follows the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) pattern: sequential `Runner.run` calls with structured outputs. The **deterministic SQL validator** (sqlglot) is the hard gate before execution; LLM steps do not replace that check.

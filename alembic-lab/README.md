# Alembic Lab

This folder contains database schema migrations for the `pe-orgair-platform` project using Alembic.

## What Is Alembic?

Alembic is a schema migration tool for SQLAlchemy-based projects.  
It tracks database structure changes over time using versioned migration files, so schema updates are:

- repeatable
- reversible (when `downgrade()` is implemented)
- consistent across environments (local, test, prod)

Instead of manually running SQL every time the schema changes, Alembic applies ordered migration scripts and records the currently applied version in the database (`alembic_version` table).

## How Alembic Works (Basic Flow)

1. You create a migration script (`alembic revision ...`).
2. The script contains two functions:
   - `upgrade()` for applying changes
   - `downgrade()` for reverting changes
3. Each migration has a `revision` id and a `down_revision` (its parent).
4. Alembic builds a revision chain/graph from these ids.
5. When you run `alembic upgrade head`, Alembic:
   - checks current DB revision
   - finds pending migrations
   - executes each `upgrade()` in order
   - updates `alembic_version`

In this project, migrations mostly use raw SQL via `op.execute(...)`, but Alembic also supports SQLAlchemy operations like `op.create_table(...)`.

## Location

- Alembic config: `../alembic.ini`
- Migration environment: `./migrations/env.py`
- Migration scripts: `./migrations/versions/`

## Prerequisites

- Python `>=3.12`
- PostgreSQL running locally
- Project dependencies installed (Alembic + SQLAlchemy + psycopg2)

You can install dependencies from the project root with Poetry:

```bash
poetry install
```

## Database Configuration

The active connection string is set in `alembic.ini`:

```ini
sqlalchemy.url = postgresql://pe_admin:pe_local_dev_123@localhost/pe_orgair_db
```

Update this value for your local environment before running migrations if needed.

## Common Commands

Run these commands from the project root (`pe-orgair-platform`), where `alembic.ini` exists.

1. Show current revision:

```bash
poetry run alembic current
```

2. Show migration history:

```bash
poetry run alembic history
```

3. Apply all pending migrations:

```bash
poetry run alembic upgrade head
```

4. Roll back one migration:

```bash
poetry run alembic downgrade -1
```

5. Create a new migration:

```bash
poetry run alembic revision -m "describe_change"
```

6. Create an autogenerate migration:

```bash
poetry run alembic revision --autogenerate -m "sync_models"
```

## Existing Revision Chain

- `ffa7dce781be` - create `sample_table`
- `dadf308f3867` - create base platform tables
- `0ee619f3516b` - seed initial focus groups/dimensions/weights data

Current head revision: `0ee619f3516b`

## Notes

- Migration scripts in this repo mostly use `op.execute(...)` with raw SQL for DDL and seed data.
- Keep migration files immutable after they are applied in shared environments.
- Prefer adding new migration files instead of editing old applied migrations.

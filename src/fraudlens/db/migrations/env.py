"""Alembic environment — async SQLAlchemy, autogenerate from ORM models."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import Base and all models so autogenerate can detect them.
import fraudlens.db.models  # noqa: F401 — registers Decision on Base.metadata
from fraudlens.core.config import get_settings
from fraudlens.db.session import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the real DB URL from Settings so alembic.ini stays secret-free.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata

# Only manage tables that belong to FraudLens (defined in Base.metadata).
_managed_tables: frozenset[str] = frozenset(target_metadata.tables.keys())


def include_object(obj: Any, name: str, type_: str, reflected: bool, compare_to: Any) -> bool:
    """Filter autogenerate to FraudLens-owned tables only."""
    if type_ == "table":
        return name in _managed_tables
    return True


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (used for dry-run diffs)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Any) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live async connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
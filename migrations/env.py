"""Alembic environment for async migrations."""
import asyncio
import sys
from pathlib import Path

# Add project root so "src" is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.storage.models import Base

config = context.config

# Set SQLAlchemy URL from application settings (ROOT-MIG-01: do not leave invalid URL)
import logging
_log = logging.getLogger(__name__)
try:
    from src.core.config import ensure_asyncpg_url, get_settings

    settings = get_settings()
    config.set_main_option(
        "sqlalchemy.url", ensure_asyncpg_url(settings.database.postgres_url)
    )
except Exception as e:
    _log.warning(
        "Could not load database URL from settings: %s. Set DATABASE__POSTGRES_URL or fix config.",
        e,
    )
    config.set_main_option(
        "sqlalchemy.url",
        "postgresql+asyncpg://localhost/memory",  # explicit placeholder; will fail if used
    )

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

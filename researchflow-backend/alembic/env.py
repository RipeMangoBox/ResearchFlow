import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from backend.database import Base
# Import all models so they register with Base.metadata
import backend.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Use DATABASE_URL_SYNC from env if available (for Docker compatibility)
db_url = os.getenv(
    "DATABASE_URL_SYNC",
    config.get_main_option("sqlalchemy.url"),
)


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(db_url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

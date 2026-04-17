.PHONY: up down db migrate migrate-down shell test

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# Start only database services (for local development)
db:
	docker compose up -d postgres redis

# Run alembic migrations
migrate:
	alembic upgrade head

# Rollback one migration
migrate-down:
	alembic downgrade -1

# Open psql shell
shell:
	docker compose exec postgres psql -U rf -d researchflow

# Run tests
test:
	pytest tests/ -v

# Run CSV -> DB migration
migrate-csv:
	python -m migration.migrate_csv_to_db

# Run MD -> DB migration
migrate-md:
	python -m migration.migrate_md_to_db

# Validate migration
validate:
	python -m migration.validate_migration

# Export DB -> Markdown
export-md:
	python -m compatibility.export_analysis_md

# Export DB -> CSV
export-csv:
	python -m compatibility.export_csv_log

# Full migration pipeline
migrate-all: migrate migrate-csv migrate-md validate
	@echo "Migration complete. Run 'make export-md' to verify round-trip."

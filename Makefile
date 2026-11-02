.DEFAULT_GOAL := help
.PHONY: help install hooks lint format typecheck test cov check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Create the virtual environment with all dev tools
	uv sync --all-extras

hooks: ## Install the pre-commit hooks
	uv run pre-commit install

lint: ## Run the linter
	uv run ruff check .

format: ## Format the code and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy (strict) on src/
	uv run mypy

test: ## Run the test suite
	uv run pytest

cov: ## Run the test suite with coverage
	uv run pytest --cov --cov-report=term-missing

check: ## Everything CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +

# To use these commands run "make <command>" in your terminal.

init:
	uv venv
	uv sync
	uv run pre-commit install
	@echo "Project initialised successfully."

update:
	uv run pre-commit autoupdate
	@echo "Pre-commit hooks are updated to the latest versions."


# Manual commands (they run as pre-commit hooks upon commit)

lint:
	uv run ruff check .
	@echo "Linting completed."

format:
	uv run ruff check --fix .
	uv run ruff format .
	@echo "Code formatting completed."

check:
	uv run pre-commit run --all-files
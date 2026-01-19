init:
	uv venv
	uv sync
	pre-commit install
	@echo "Project initialized. Please restart your shell to activate the virtual environment."

update:  # could include more update steps in the future
	pre-commit autoupdate
	pre-commit run --all-files
	@echo "Pre-commit hooks updated and run on all files."
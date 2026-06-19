# Vats_up

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.

### Install uv (if not installed already)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Common commands

```bash
# Create a virtual environment and install dependencies from pyproject.toml / uv.lock
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run a Python script inside the project environment
uv run python main.py

# Run an arbitrary command (e.g. the submission validator)
uv run python scripts/validate_submission.py

# Update the lockfile
uv lock

# Upgrade all dependencies to the latest allowed versions
uv sync --upgrade

# Show the active Python interpreter
uv run python -V
```

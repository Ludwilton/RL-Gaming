# RL-Gaming
*This is a temporary README that will get updated for the duration of the project.*

## Installation

1. Clone the repo
2. Install UV globally on your system
3. Open the repo and run "make init"

## Development

### Ruff

It's recommended to install Ruff as an extension to your IDE - if it exists. In VS Code you will see feedback from Ruff in the "Problems" tab (same window as your integrated terminal).

To get the same functionality manually you will need to run:

```bash
uv run ruff check .  # lint all files without fixes
uv run ruff check --fix .  # lint all files with automatic (safe) fixes
uv run ruff format .  # format all files
```

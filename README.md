# Reinforcement Learning with MiniGrid
This project is part of a deep learning course at [ITHS](https://www.iths.se), taught by [Raphael Korsoski](https://www.github.com/pr0fez). The purpose of the project is to deep dive into a field of our choice within deep learning.

For more information regarding this specific project, see [project definition](documentation/01_project_definition.ipynb).

**Disclaimer:** *This is a work in progress.*

## Installation

1. Clone the repo
2. Install UV globally on your system
3. Open the repo and run `make init`

## Development

### Ruff

It's recommended to install Ruff as an extension to your IDE - if it exists. In VS Code you will see feedback from Ruff in the "Problems" tab (same window as your integrated terminal).

To get the same functionality manually you will need to run:

```bash
uv run ruff check .  # lint all files without fixes
uv run ruff check --fix .  # lint all files with automatic (safe) fixes
uv run ruff format .  # format all files
```

## Project structure

```
.
├── .github/
│   └── movielens/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
├── data/
│   └── ppo/
│   │   ├── level_4_TS1000k_see_through_walls_model.zip
│   │   └── monitor.csv
│   ├── rppo/
│   │   ├── model.zip
│   │   ├── monitor.csv
│   │   └── values.json
│   └── rppo_2/
│   │   ├── monitor.csv
│   │   └── recurrentppo256dim.zip
├── documentation/
│   └── 01_project_definition.ipynb
├── src/
│   └── rl_gaming/
│   │   ├── __init__.py
│   │   ├── custom_callback.py
│   │   ├── env_optimiser.py
│   │   ├── feature_extractor.py
│   │   ├── levels_extended.json
│   │   ├── levels_test.ipynb
│   │   ├── levels_utils.py
│   │   ├── levels.json
│   │   ├── minigrid_levels_env.py
│   │   ├── minigrid_onehot.py
│   │   ├── model_utils.py
│   │   ├── plot_utils.py
│   │   ├── presentation.ipynb
│   │   ├── procedural_level.py
│   │   ├── simple_env.py
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── conventional_commits.md
├── LICENSE
├── Makefile
├── onehotrecurrentPPO1000difficulty400ksteps.mp4
├── pyproject.toml
├── README.md
├── uv.lock
```
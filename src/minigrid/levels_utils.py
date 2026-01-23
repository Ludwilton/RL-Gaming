import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LEVELS_PATH = BASE_DIR / "levels.json"


def get_levels() -> dict:
    """Get all levels from json file."""
    with LEVELS_PATH.open(encoding="utf-8") as f:
        levels = json.load(f)
        return {int(k): v for k, v in levels.items()}

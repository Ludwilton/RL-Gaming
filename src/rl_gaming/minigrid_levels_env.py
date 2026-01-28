from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from gymnasium.spaces import Discrete
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

BASE_DIR = Path(__file__).resolve().parent
LEVELS_PATH = BASE_DIR / "levels.json"


class MiniGridLevelsEnv(MiniGridEnv):
    """MiniGrid levels environment class."""

    def __init__(
        self,
        level_id: int | None = None,
        levels: list | None = None,
        see_through_walls: bool = False,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the environment.

        Args:
            level_id (int): ID of a single level for environment.
            levels (list): List of levels for environment to select random from.
            see_through_walls (bool): Setting for if the agent should see through walls.
            render_mode (str): Render mode of the environment.
            **kwargs: Additional keyword arguments.

        """
        mission_space = MissionSpace(mission_func=lambda: "Level Environment")

        levels_config = self.get_levels()

        self.levels = levels if levels is not None else list(levels_config.keys())
        self.fixed_level = level_id is not None

        height, width = 0, 0
        if self.fixed_level:
            height, width = self._setup_level(level_id)
        else:
            height, width = self._setup_level(random.choice(self.levels))

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=see_through_walls,
            agent_view_size=7,
            render_mode=render_mode,
            **kwargs,
        )

        # Allow only 3 actions permitted: left, right, forward
        # (Available actions in 'minigrid.core.actions')
        self.action_space = Discrete(3)

    @classmethod
    def get_levels(cls) -> dict:
        """Get all levels from json file."""
        with LEVELS_PATH.open(encoding="utf-8") as f:
            levels = json.load(f)
            return {int(k): v for k, v in levels.items()}

    def reset(self, **kwargs: dict[str, Any]) -> tuple:
        """Reset environment."""
        if not self.fixed_level:
            new_level = random.choice(self.levels)
            self._setup_level(new_level)

        return super().reset(**kwargs)

    def _setup_level(self, level_id: int | None) -> tuple:
        """Set up level."""
        levels_config = self.get_levels()

        self.level_id = level_id
        self.level = levels_config[level_id]

        agent_config = self.level.get("agent", {})
        self.agent_start_pos = agent_config.get("pos")
        self.agent_start_dir = agent_config.get("dir")

        height = self.level.get("height", self.level.get("size", 5))
        width = self.level.get("width", self.level.get("size", 5))

        max_value = max(height, width)
        self.max_steps = 4 * max_value**2

        self.max_steps = min(100, self.max_steps)  # Max 100

        self.mission_space = "Level " + str(self.level_id)

        # Environment configuration
        self.width = width
        self.height = height

        # Current grid and mission and carrying
        self.grid = Grid(width, height)

        return height, width

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate level grid.

        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.

        """
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Walls
        if "walls" in self.level:
            for wall in self.level.get("walls", []):
                if wall["type"] == "rect":
                    self.grid.wall_rect(wall["x"], wall["y"], wall["w"], wall["h"])
                elif wall["type"] == "square":
                    self.put_obj(Wall(), wall["x"], wall["y"])

        # Lava
        if "lava" in self.level:
            for lava in self.level.get("lava", []):
                self.put_obj(Lava(), lava["x"], lava["y"])

        # Goal
        self.put_obj(Goal(), *self.level["goal"])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = tuple(self.agent_start_pos)

        if self.agent_start_dir is not None:
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_dir = self._rand_int(0, 4)


def human_test_level(level_id: int = 1) -> None:
    """Test the MiniGrid environment with manual control."""
    env = MiniGridLevelsEnv(render_mode="human", level_id=level_id)

    # Test level with manual control
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    human_test_level(level_id=6)

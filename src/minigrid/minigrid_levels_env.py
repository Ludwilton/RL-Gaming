from __future__ import annotations

import json
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
        level_id: int = 1,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the environment.

        Args:
            level_id (int): ID of the level for environment.
            render_mode (str): Render mode of the environment.
            **kwargs: Additional keyword arguments.

        """
        levels = self.get_levels()

        self.level_id = level_id
        self.level = levels[level_id]

        agent_config: dict[str, Any] = self.level.get("agent", {})
        self.agent_start_pos = agent_config.get("pos")
        self.agent_start_dir = agent_config.get("dir")

        mission_space = MissionSpace(mission_func=lambda: "Level " + str(self.level_id))

        height = 5
        width = 5
        if "height" in self.level and "width" in self.level:
            height = self.level["height"]
            width = self.level["width"]
        elif "size" in self.level:
            height = self.level["size"]
            width = self.level["size"]

        max_value = max(height, width)

        max_steps = 4 * max_value**2

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            see_through_walls=True,
            max_steps=max_steps,
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

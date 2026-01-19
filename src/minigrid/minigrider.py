from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl


LEVELS = {
    1: {
        "width": 5,
        "height": 5,
        "goal": (3, 3),
        "agent_start": (1, 1),
    },

    2: {
        "width": 7,
        "height": 7,
        "goal": (5, 5),
        "agent_start": (1, 1),
    },

    3: {
        "width": 14,
        "height": 14,
        "goal": (2, 3),
        "agent_start": (12, 12),
    }
}

class MiniGrider(MiniGridEnv):
    LEVELS = LEVELS  # expose as class variable

    @classmethod
    def get_available_levels(cls):
        return sorted(cls.LEVELS.keys())

    def __init__(self, level_id=1, **kwargs):
        self.level_id = level_id
        self.level = LEVELS[level_id]

        """Initialize a new environment."""
        self.agent_start_pos = self.level["agent_start"]
        self.agent_start_dir = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

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
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Walls
        if "walls" in self.level:
            for wall in self.level["walls"]:
                if wall["type"] == "vertical":
                    for y in range(height):
                        self.grid.set(wall["x"], y, Wall())
                elif wall["type"] == "horizontal":
                    for x in range(width):
                        self.grid.set(x, wall["y"], Wall())

        # Doors
        if "doors" in self.level:
            for door in self.level["doors"]:
                self.grid.set(
                    *door["pos"],
                    Door(door["color"], is_locked=door["locked"])
                )

        # Keys
        if "keys" in self.level:
            for key in self.level["keys"]:
                self.grid.set(
                    *key["pos"],
                    Key(key["color"])
                )

        # Goal
        self.put_obj(Goal(), *self.level["goal"])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Level " + str(self.level_id)



def human_test() -> None:
    env = MiniGrider(render_mode="human", level_id=3)

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    human_test()

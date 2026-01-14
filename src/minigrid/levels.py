from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class LevelOne(MiniGridEnv):
    """Empty grid environment, no obstacles, sparse reward."""

    def __init__(
        self,
        size: int = 16,
        agent_start_pos: set[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs: dict[str, dict],
    ) -> None:
        """Initialize a new LevelOne environment."""
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,  # set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Level 1"

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Level 1"


class LevelTwo(LevelOne):
    """Empty grid with lava pool in optimal path."""

    @staticmethod
    def _gen_mission() -> str:
        return "avoid the lava and get to the green goal square"

    def _gen_grid(self, width: int, height: int) -> None:
        # Reuse parent's grid generation
        super()._gen_grid(width, height)

        # Add lava pools
        lava_coordinates = [
            (1, 3),
            (3, 1),
            (14, 12),
            (12, 14),
        ]
        [self.grid.set(x, y, Lava()) for x, y in lava_coordinates]


# FIXME: remove this test code later
def main() -> None:
    """Run a simple test of LevelTwo environment."""
    env = LevelTwo(render_mode="human")
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()

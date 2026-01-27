from __future__ import annotations

from collections import deque

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


class ProceduralLevel(MiniGridEnv):
    """random generated level, walls & lava, difficulty .

    Parameters
    ----------
        size: size of the grid (size x size) - walls included
        max_steps: maximum number of steps per episode
        agent_start_pos: starting position of the agent (x, y)
        agent_start_dir: starting direction of the agent (0: right, 1: down, 2: left, 3: up)
        difficulty: difficulty level from 0 (easy) to 1000 (hard)
    returns:
        MiniGrid environment with procedurally generated levels.

    """

    def __init__(
        self,
        size: int = 10,
        max_steps: int | None = None,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        difficulty: int = 0,  # scale 0-1000
        **kwargs: dict[str, dict],
    ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.difficulty = max(0, min(1000, difficulty))

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        # (Available actions in 'minigrid.core.actions')
        self.action_space = Discrete(3)

    @staticmethod
    def _gen_mission() -> str:
        return "procedurally generated level"

    def _is_solvable(self, width: int, height: int) -> bool:
        goal_pos = None
        for x in range(width):
            for y in range(height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Goal):
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        if not goal_pos:
            return False

        queue = deque([self.agent_pos])
        visited = {self.agent_pos}

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal_pos:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                if 0 <= nx < width and 0 <= ny < height:
                    cell = self.grid.get(nx, ny)
                    if cell is None or isinstance(cell, Goal):
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False

    def _gen_grid(self, width: int, height: int) -> None:
        max_attempts = 100

        for _ in range(max_attempts):
            self.grid = Grid(width, height)

            self.grid.wall_rect(0, 0, width, height)
            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.place_agent()

            available_space = (width - 2) * (height - 2) - 2

            num_walls = int((self.difficulty / 1000) * min(30, available_space * 0.15))
            num_lava = int((self.difficulty / 1000) * min(20, available_space * 0.10))

            for _ in range(num_walls):
                self.place_obj(Wall(), max_tries=100)

            for _ in range(num_lava):
                self.place_obj(Lava(), max_tries=100)

            self.place_obj(Goal(), max_tries=100)

            if self._is_solvable(width, height):
                self.mission = (
                    f"procedurally generated level (difficulty: {self.difficulty})"
                )
                return

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.place_obj(Goal(), max_tries=100)

        self.mission = f"procedurally generated level (difficulty: {self.difficulty})"


def main() -> None:
    """Run a simple test of environment."""
    env = ProceduralLevel(render_mode="human", difficulty=1000)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()

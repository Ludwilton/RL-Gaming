from __future__ import annotations

from collections import deque

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
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
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Level 1"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), width - 2, height - 2)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Level 1"


class LevelTwo(LevelOne):
    @staticmethod
    def _gen_mission() -> str:
        return "avoid the lava and get to the green goal square"

    def _gen_grid(self, width: int, height: int) -> None:
        super()._gen_grid(width, height)

        lava_coordinates = [
            (1, 3),
            (3, 1),
            (14, 12),
            (12, 14),
        ]
        [self.grid.set(x, y, Lava()) for x, y in lava_coordinates]


class ProceduralLevel(MiniGridEnv):
    """
    random generated level, walls & lava, difficulty scaling
    """
    def __init__(
        self,
        size: int = 10,
        max_steps: int | None = None,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        difficulty: int = 0, #scale 0-1000
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
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

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

        for attempt in range(max_attempts):
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
                self.mission = f"procedurally generated level (difficulty: {self.difficulty})"
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


# FIXME: remove this test code later
def main() -> None:
    """Run a simple test of LevelTwo environment."""
    env = ProceduralLevel(render_mode="human", difficulty=800)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()

"""Maze tasks that are defined by their map, termination condition, and goals.
"""
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type
import numpy as np
from utils.cv_utils import blob_detect
from simulations.maze_env_utils import MazeCell
import copy

class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"

RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)

def get_hsv_ranges(rgb):
    if rgb == RED:
        return (0, 25, 0), (15, 255, 255)
    elif rgb == GREEN:
        return (36,0,0), (86,255,255)
    elif rgb == BLUE:
        return (94, 80, 2), (126, 255, 255)

class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 1.0,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float:
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0, 4.0)
    INNER_REWARD_SCALING: float = 0.01
    # For Fall/Push/BlockMaze
    # For Billiar   
    OBJECT_BALL_SIZE: float = 1.0
    # Unused now
    PUT_SPIN_NEAR_AGENT: bool = False

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale


    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class DistRewardMixIn:
    REWARD_THRESHOLD: float = -1000.0
    goals: List[MazeGoal]
    scale: float

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs) / self.scale


class GoalRewardUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]


    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardUMaze(GoalRewardUMaze, DistRewardMixIn):
    pass


class GoalRewardSimpleRoom(GoalRewardUMaze):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([2.0 * scale, 0.0]))]
    

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardSimpleRoom(GoalRewardSimpleRoom, DistRewardMixIn):
    pass


class GoalRewardSquareRoom(GoalRewardUMaze):
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 2.0)

    def __init__(self, scale: float, goal: Tuple[float, float] = (1.0, 0.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]


    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B],
            [B, E, E, E, B],
            [B, E, R, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class NoRewardSquareRoom(GoalRewardSimpleRoom):
    def __init__(self, scale: float) -> None:
        super().__init__(scale)

    def reward(self, _obs: np.ndarray) -> float:
        return 0.0


class DistRewardSquareRoom(GoalRewardSquareRoom, DistRewardMixIn):
    pass


class GoalRewardPush(GoalRewardUMaze):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.375 * scale]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R, M = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.XY_BLOCK
        return [
            [B, B, B, B, B],
            [B, E, R, B, B],
            [B, E, M, E, B],
            [B, B, E, B, B],
            [B, B, B, B, B],
        ]


class DistRewardPush(GoalRewardPush, DistRewardMixIn):
    pass


class GoalRewardFall(GoalRewardUMaze):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.375 * scale, 4.5]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, C, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.CHASM, MazeCell.ROBOT
        M = MazeCell.YZ_BLOCK
        return [
            [B, B, B, B],
            [B, R, E, B],
            [B, E, M, B],
            [B, C, C, B],
            [B, E, E, B],
            [B, B, B, B],
        ]


class DistRewardFall(GoalRewardFall, DistRewardMixIn):
    pass


class GoalReward2Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[int, int] = (4.0, -2.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, B],
            [B, E, E, E, B, E, E, B],
            [B, E, R, E, B, E, E, B],
            [B, E, E, E, B, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B],
        ]


class DistReward2Rooms(GoalReward2Rooms, DistRewardMixIn):
    pass


class SubGoal2Rooms(GoalReward2Rooms):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (4.0, -2.0),
        subgoals: List[Tuple[float, float]] = [(1.0, -2.0), (-1.0, 2.0)],
    ) -> None:
        super().__init__(scale, primary_goal)
        for subgoal in subgoals:
            self.goals.append(
                MazeGoal(np.array(subgoal) * scale, reward_scale=0.5, rgb=GREEN)
            )


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[int, int] = (6.0, -6.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class DistReward4Rooms(GoalReward4Rooms, DistRewardMixIn):
    pass

class MazeVisualGoal(MazeGoal):
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 1.5,
        custom_size: Optional[float] = None,
    ):
        super(MazeVisualGoal, self).__init__(pos,
            reward_scale,
            rgb,
            threshold,
            custom_size,
        )
        self.min_range, self.max_range = get_hsv_ranges(rgb)

    def neighbor(self, pos: np.ndarray) -> float:
        return np.linalg.norm(pos[: self.dim] - self.pos) <= self.threshold

    def inframe(self, obs):
        out = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        #cv2.imshow('stream', out)
        #cv2.waitKey(1)
        keypoints, _ = blob_detect(
            out,
            self.min_range,
            self.max_range
        )
        if len(keypoints) == 0:
            return False
        return True

class CustomGoalReward4Rooms(GoalReward4Rooms):
    def __init__(self,
        scale: float,
        goal: Tuple[int, int] = (6.0, -6.0),
        danger: List[Tuple[int, int]] = [
            (0.0, -6.0),
            (6.0, 0.0),
        ]) -> None:
        super().__init__(scale, goal)
        self.set()

    def set(self):
        self.goal_index = np.random.randint(low = 0, high = 3)
        self.colors = []
        self.scales = []
        self.goals = []
        for i in range(3):
            if i == self.goal_index:
                self.colors.append(copy.deepcopy(RED))
                self.scales.append(1.0)
            else:
                self.colors.append(copy.deepcopy(GREEN))
                self.scales.append(0.5)

        self.goals = [ 
            MazeVisualGoal(np.array([
                np.random.uniform(4.0, 6.0) * self.scale,
                np.random.uniform(-6.0, -4.0) * self.scale
            ]), self.scales[0], self.colors[0]),
            MazeVisualGoal(np.array([
                np.random.uniform(0.0, 2.0) * self.scale,
                np.random.uniform(-6.0, -4.0) * self.scale
            ]), self.scales[1], self.colors[1]),
            MazeVisualGoal(np.array([
                np.random.uniform(4.0, 6.0) * self.scale,
                np.random.uniform(-2.0, 0.0) * self.scale
            ]), self.scales[2], self.colors[2]),
        ]

    def reward(self, obs: np.ndarray, pos: np.ndarray) -> float:
        reward = 0.0
        for i, goal in enumerate(self.goals):
            scale = -1.0
            if i == self.goal_index:
                scale = 1.0
            if goal.inframe(obs):
                reward += goal.reward_scale * scale + np.abs(scale) * (
                    1 - np.linalg.norm(pos[: goal.dim] - goal.pos) / (np.linalg.norm(goal.pos))
                )
        return reward

    def termination(self, pos: np.ndarray) -> bool:
        if self.goals[self.goal_index].neighbor(pos):
            return True
        return False


class GoalRewardTRoom(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)

    def __init__(self, scale: float, goal: Tuple[float, float] = (2.0, -3.0)) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array(goal) * scale)]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, B, E, E, B],
            [B, E, E, B, E, E, B],
            [B, E, B, B, B, E, B],
            [B, E, E, R, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardTRoom(GoalRewardTRoom, DistRewardMixIn):
    pass


class SubGoalTRoom(GoalRewardTRoom):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (2.0, -3.0),
        subgoal: Tuple[float, float] = (-2.0, -3.0),
    ) -> None:
        super().__init__(scale, primary_goal)
        self.goals.append(
            MazeGoal(np.array(subgoal) * scale, reward_scale=0.5, rgb=GREEN)
        )


class NoRewardCorridor(MazeTask):
    REWARD_THRESHOLD: float = 0.0
    MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 1.0)

    def reward(self, _obs: np.ndarray) -> float:
        return 0.0

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, B, E, E, E, E, B],
            [B, E, E, B, E, E, E, E, B],
            [B, E, E, E, E, E, B, B, B],
            [B, E, E, E, R, E, E, E, B],
            [B, B, B, E, E, E, E, E, B],
            [B, E, E, E, E, B, E, E, B],
            [B, E, E, E, E, B, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class GoalRewardCorridor(NoRewardCorridor):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001

    def __init__(self, scale: float, goal: Tuple[float, float] = (3.0, -3.0)) -> None:
        super().__init__(scale)
        self.goals.append(MazeGoal(np.array(goal) * scale))

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY


class DistRewardCorridor(GoalRewardCorridor, DistRewardMixIn):
    pass


class GoalRewardBlockMaze(GoalRewardUMaze):
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0, None)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 3.0 * scale]))]

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        M = MazeCell.XY_BLOCK
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, M, B],
            [B, E, E, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]


class DistRewardBlockMaze(GoalRewardBlockMaze, DistRewardMixIn):
    pass


class GoalRewardBilliard(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(None, 3.0, None)
    GOAL_SIZE: float = 0.3

    def __init__(self, scale: float, goal: Tuple[float, float] = (2.0, -3.0)) -> None:
        super().__init__(scale)
        goal = np.array(goal) * scale
        self.goals.append(
            MazeGoal(goal, threshold=self._threshold(), custom_size=self.GOAL_SIZE)
        )

    def _threshold(self) -> float:
        return self.OBJECT_BALL_SIZE + self.GOAL_SIZE

    def reward(self, obs: np.ndarray) -> float:
        object_pos = obs[3:6]
        for goal in self.goals:
            if goal.neighbor(object_pos):
                return goal.reward_scale
        return self.PENALTY

    def termination(self, obs: np.ndarray) -> bool:
        object_pos = obs[3:6]
        for goal in self.goals:
            if goal.neighbor(object_pos):
                return True
        return False

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B = MazeCell.EMPTY, MazeCell.BLOCK
        R, M = MazeCell.ROBOT, MazeCell.OBJECT_BALL
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, E, E, M, E, E, B],
            [B, E, E, R, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class DistRewardBilliard(GoalRewardBilliard):
    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs[3:6]) / self.scale


class NoRewardBilliard(GoalRewardBilliard):
    def __init__(self, scale: float) -> None:
        MazeTask.__init__(self, scale)

    def reward(self, _obs: np.ndarray) -> float:
        return 0.0


class SubGoalBilliard(GoalRewardBilliard):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (2.0, -3.0),
        subgoals: List[Tuple[float, float]] = [(-2.0, -3.0), (-2.0, 1.0), (2.0, 1.0)],
    ) -> None:
        super().__init__(scale, primary_goal)
        for subgoal in subgoals:
            self.goals.append(
                MazeGoal(
                    np.array(subgoal) * scale,
                    reward_scale=0.5,
                    rgb=GREEN,
                    threshold=self._threshold(),
                    custom_size=self.GOAL_SIZE,
                )
            )


class BanditBilliard(SubGoalBilliard):
    def __init__(
        self,
        scale: float,
        primary_goal: Tuple[float, float] = (4.0, -2.0),
        subgoals: List[Tuple[float, float]] = [(4.0, 2.0)],
    ) -> None:
        super().__init__(scale, primary_goal, subgoals)

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B = MazeCell.EMPTY, MazeCell.BLOCK
        R, M = MazeCell.ROBOT, MazeCell.OBJECT_BALL
        return [
            [B, B, B, B, B, B, B],
            [B, E, E, B, B, E, B],
            [B, E, E, E, E, E, B],
            [B, R, M, E, B, B, B],
            [B, E, E, E, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]


class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "SimpleRoom": [DistRewardSimpleRoom, GoalRewardSimpleRoom],
        "SquareRoom": [DistRewardSquareRoom, GoalRewardSquareRoom, NoRewardSquareRoom],
        "UMaze": [DistRewardUMaze, GoalRewardUMaze],
        "Push": [DistRewardPush, GoalRewardPush],
        "Fall": [DistRewardFall, GoalRewardFall],
        "2Rooms": [DistReward2Rooms, GoalReward2Rooms, SubGoal2Rooms],
        "4Rooms": [DistReward4Rooms, GoalReward4Rooms, CustomGoalReward4Rooms],
        "TRoom": [DistRewardTRoom, GoalRewardTRoom, SubGoalTRoom],
        "BlockMaze": [DistRewardBlockMaze, GoalRewardBlockMaze],
        "Corridor": [DistRewardCorridor, GoalRewardCorridor, NoRewardCorridor],
        "Billiard": [
            DistRewardBilliard,  # v0
            GoalRewardBilliard,  # v1
            SubGoalBilliard,  # v2
            BanditBilliard,  # v3
            NoRewardBilliard,  # v4
        ],
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]

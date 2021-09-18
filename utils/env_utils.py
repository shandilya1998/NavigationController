import itertools as it
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

class MazeCell(Enum):
    # Robot: Start position
    ROBOT = -1
    # Blocks
    EMPTY = 0
    BLOCK = 1
    CHASM = 2
    OBJECT_BALL = 3
    # Moves
    XY_BLOCK = 14
    XZ_BLOCK = 15
    YZ_BLOCK = 16
    XYZ_BLOCK = 17
    XY_HALF_BLOCK = 18
    SPIN = 19

    def is_block(self) -> bool:
        return self == self.BLOCK

    def is_chasm(self) -> bool:
        return self == self.CHASM

    def is_object_ball(self) -> bool:
        return self == self.OBJECT_BALL

    def is_empty(self) -> bool:
        return self == self.ROBOT or self == self.EMPTY

    def is_robot(self) -> bool:
        return self == self.ROBOT

    def is_wall_or_chasm(self) -> bool:
        return self in [self.BLOCK, self.CHASM]

    def can_move_x(self) -> bool:
        return self in [
            self.XY_BLOCK,
            self.XY_HALF_BLOCK,
            self.XZ_BLOCK,
            self.XYZ_BLOCK,
            self.SPIN,
        ]

    def can_move_y(self) -> bool:
        return self in [
            self.XY_BLOCK,
            self.XY_HALF_BLOCK,
            self.YZ_BLOCK,
            self.XYZ_BLOCK,
            self.SPIN,
        ]

    def can_move_z(self) -> bool:
        return self in [self.XZ_BLOCK, self.YZ_BLOCK, self.XYZ_BLOCK]

    def can_spin(self) -> bool:
        return self == self.SPIN

    def can_move(self) -> bool::
        return self.can_move_x() or self.can_move_y() or self.can_move_z()

    def is_half_block(self) -> bool:
        return self in [self.XY_HALF_BLOCK]


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 0.6,
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
        """
            Need to modify this to fit an image observation
        """
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5

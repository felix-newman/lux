from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class RewardedActionType(Enum):
    MINE_ICE = 1
    MINE_ORE = 2
    TRANSFER_ICE = 3
    TRANSFER_ORE = 4
    PICKUP_POWER = 5

    def to_action_type(self):
        return ActionType.__members__.get(str(self.value))


class ActionType(Enum):
    MINE_ICE = 1
    MINE_ORE = 2
    TRANSFER_ICE = 3
    TRANSFER_ORE = 4
    PICKUP_POWER = 5
    MOVE_CENTER = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_RIGHT = 3
    MOVE_LEFT = 4

    @property
    def is_factory_action(self) -> bool:
        if self.value in [3, 4, 5]:
            return True
        return False

    @property
    def is_rewarded_action(self) -> bool:
        if RewardedActionType.__members__.get(str(self.value)) is None:
            return False
        return True

    def to_rewarded_action_type(self):
        return RewardedActionType.__members__.get(str(self.value))


class Direction(Enum):
    CENTER = 0
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


@dataclass
class ActionItem:
    type: ActionType
    direction: Direction
    repeat: int
    position: np.array


@dataclass
class ActionSequence:
    action_items: List[ActionItem]
    remaining_rewards: List[int]
    reward: int = 0

    def to_lux_action_queue(self) -> List[np.array]:
        return []


@dataclass
class RewardedActionSequence:
    action_items: List[RewardedActionType]

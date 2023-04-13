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


class ActionType(Enum):
    MOVE_CENTER = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_RIGHT = 3
    MOVE_LEFT = 4
    TRANSFER_ICE = 5
    TRANSFER_ORE = 6
    MINE_ICE = 7
    MINCE_ORE = 8
    PICKUP_POWER = 9


@dataclass
class ActionItem:
    type: ActionType
    repeat: int
    position: np.array


@dataclass
class ActionSequence:
    action_items: List[ActionItem]
    reward: int = 0

    def to_lux_action_queue(self) -> List[np.array]:
        return []


@dataclass
class RewardedActionSequence:
    action_items: List[RewardedActionType]

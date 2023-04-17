from dataclasses import dataclass
from enum import Enum
from typing import List, Set
from typing_extensions import Literal

import numpy as np


class ActionType(Enum):
    MINE_ICE = 0
    MINE_ORE = 1
    TRANSFER_ICE = 2
    TRANSFER_ORE = 3
    PICKUP_POWER = 4
    MOVE_CENTER = 5
    MOVE_UP = 6
    MOVE_DOWN = 7
    MOVE_RIGHT = 8
    MOVE_LEFT = 9

    @property
    def is_factory_action(self) -> bool:
        if self in factory_actions:
            return True
        return False

    @property
    def is_rewarded_action(self) -> bool:
        if self in rewarded_actions:
            return True
        return False


factory_actions: Set[ActionType] = {ActionType.TRANSFER_ICE, ActionType.TRANSFER_ORE, ActionType.PICKUP_POWER}
rewarded_actions: Set[ActionType] = {ActionType.MINE_ICE, ActionType.MINE_ORE, ActionType.TRANSFER_ICE, ActionType.TRANSFER_ORE,
                                     ActionType.PICKUP_POWER}

RewardedAction = Literal[ActionType.MINE_ICE, ActionType.MINE_ORE, ActionType.TRANSFER_ICE, ActionType.TRANSFER_ORE,
ActionType.PICKUP_POWER]


class Direction(Enum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


@dataclass
class ActionItem:
    type: ActionType
    direction: Direction
    repeat: int
    amount: int
    position: np.array

    def to_lux_action(self) -> np.array:
        if self.type == ActionType.MOVE_CENTER:
            return np.array([0, 0, 0, 0, 0, self.repeat])
        if self.type == ActionType.MOVE_UP:
            return np.array([0, 1, 0, 0, 0, self.repeat])
        if self.type == ActionType.MOVE_DOWN:
            return np.array([0, 3, 0, 0, 0, self.repeat])
        if self.type == ActionType.MOVE_RIGHT:
            return np.array([0, 2, 0, 0, 0, self.repeat])
        if self.type == ActionType.MOVE_LEFT:
            return np.array([0, 4, 0, 0, 0, self.repeat])

        if self.type == ActionType.MINE_ICE:
            return np.array([3, 0, 0, 0, 0, self.repeat])
        if self.type == ActionType.MINE_ORE:
            return np.array([3, 0, 0, 0, 0, self.repeat])

        if self.type == ActionType.TRANSFER_ICE:  # TODO amount currently hardcoded
            return np.array([1, int(self.direction.value), 0, 3000, 0, self.repeat])
        if self.type == ActionType.TRANSFER_ORE:
            return np.array([1, int(self.direction.value), 1, 3000, 0, self.repeat])

        if self.type == ActionType.PICKUP_POWER:
            return np.array([2, 0, 4, self.amount, 0, self.repeat])


@dataclass
class ActionSequence:
    action_items: List[ActionItem]
    remaining_rewards: List[int]
    reward: int = 0

    def estimate_lux_action_queue_length(self) -> int:
        cur_length = 0
        if self.empty:
            return 0
        cur_item = self.action_items[0]
        for item in self.action_items[1:]:
            if item.type != cur_item.type:
                cur_length += 1
                cur_item = item
        cur_length += 1
        return cur_length

    def to_lux_action_queue(self) -> List[np.array]:
        lux_actions = []
        if self.empty:
            return []
        cur_item = self.action_items[0]
        for item in self.action_items[1:]:
            if item.type == cur_item.type:
                cur_item.repeat += item.repeat
            else:
                lux_actions.append(cur_item.to_lux_action())
                cur_item = item

        lux_actions.append(cur_item.to_lux_action())
        return lux_actions

    def step(self) -> None:
        # TODO: update remaining rewards
        if not self.empty:
            if self.action_items[0].repeat > 1:
                self.action_items[0].repeat -= 1
            else:
                _ = self.action_items.pop(0)
            return
        return

    @property
    def empty(self):
        return len(self.action_items) == 0

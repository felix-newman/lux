import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple, Union

import numpy as np
from typing_extensions import Literal


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
    DIG = 10
    RETURN = 11
    RECHARGE = 12
    FIGHT = 13
    LOOT = 14

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
move_actions: Set[ActionType] = {ActionType.MOVE_CENTER, ActionType.MOVE_UP, ActionType.MOVE_DOWN, ActionType.MOVE_RIGHT,
                                 ActionType.MOVE_LEFT}
rewarded_actions: Set[ActionType] = {ActionType.MINE_ICE, ActionType.MINE_ORE, ActionType.TRANSFER_ICE, ActionType.TRANSFER_ORE,
                                     ActionType.PICKUP_POWER, ActionType.DIG, ActionType.RETURN, ActionType.RECHARGE, ActionType.FIGHT, ActionType.LOOT}

RewardedAction = Literal[ActionType.MINE_ICE, ActionType.MINE_ORE, ActionType.TRANSFER_ICE, ActionType.TRANSFER_ORE,
ActionType.PICKUP_POWER, ActionType.DIG, ActionType.RETURN, ActionType.RECHARGE, ActionType.FIGHT, ActionType.LOOT]


class Direction(Enum):
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


DIRECTION_DELTAS = {
    Direction.CENTER: np.array([0, 0]),
    Direction.UP: np.array([0, -1]),
    Direction.RIGHT: np.array([1, 0]),
    Direction.DOWN: np.array([0, 1]),
    Direction.LEFT: np.array([-1, 0]),
}


@dataclass
class ActionItem:
    def __str__(self):
        abbrev = {
            ActionType.MINE_ICE: "MI",
            ActionType.MINE_ORE: "MO",
            ActionType.TRANSFER_ICE: "TI",
            ActionType.TRANSFER_ORE: "TO",
            ActionType.PICKUP_POWER: "PP",
            ActionType.MOVE_CENTER: "C",
            ActionType.MOVE_UP: "U",
            ActionType.MOVE_DOWN: "D",
            ActionType.MOVE_RIGHT: "R",
            ActionType.MOVE_LEFT: "L",
            ActionType.DIG: "DG",
            ActionType.RETURN: "RT",
            ActionType.RECHARGE: "RC",
            ActionType.FIGHT: "F",
            ActionType.LOOT: "LT",
        }

        return f"({abbrev[self.type]}, {self.repeat})"

    type: ActionType
    direction: Direction
    repeat: int
    amount: int
    position: np.array
    reward: float = 0.0

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
        if self.type == ActionType.DIG:
            return np.array([3, 0, 0, 0, 0, self.repeat])

        if self.type == ActionType.TRANSFER_ICE:
            return np.array([1, int(self.direction.value), 0, 3000, 0, self.repeat])
        if self.type == ActionType.TRANSFER_ORE:
            return np.array([1, int(self.direction.value), 1, 3000, 0, self.repeat])

        if self.type == ActionType.PICKUP_POWER:
            return np.array([2, 0, 4, int(self.amount), 0, self.repeat])

        if self.type == ActionType.RETURN:
            return None
        if self.type == ActionType.FIGHT:
            return None
        if self.type == ActionType.LOOT:
            return np.array([3, 0, 0, 0, 0, self.repeat])

        if self.type == ActionType.RECHARGE:
            return np.array([5, 0, 4, int(self.amount), 0, 1])


@dataclass
class ActionSequence:
    action_items: List[ActionItem]
    remaining_rewards: List[int]
    reward: int = 0

    def __str__(self):
        return "".join([str(item) for item in self.action_items])

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
        items = copy.deepcopy(self.action_items)  # create a copy of action_items
        cur_item = items.pop(0)
        for item in items:
            if item.type == cur_item.type:
                if item.type in move_actions:
                    cur_item.repeat += item.repeat
                else:
                    if item.position[0] == cur_item.position[0] and item.position[1] == cur_item.position[1]:
                        cur_item.repeat += item.repeat
                    else:
                        lux_actions.append(cur_item.to_lux_action())
                        cur_item = item
            else:
                lux_actions.append(cur_item.to_lux_action())
                cur_item = item

        lux_actions.append(cur_item.to_lux_action())
        lux_actions = [x for x in lux_actions if x is not None]
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


def rewarded_actions_from_lux_action_queue(action_queue: List[np.array]) -> List[RewardedAction]:
    def lux_action_to_type(lux_action: np.array, resource: Union[None, int]) -> Tuple[Union[None, ActionType], Union[None, int]]:
        if lux_action[0] == 3:
            if resource == 0:
                return ActionType.MINE_ICE, None
            elif resource == 1:
                return ActionType.MINE_ORE, None
            else:
                return ActionType.DIG, None
        if lux_action[0] == 1:
            if lux_action[2] == 0:
                return ActionType.TRANSFER_ICE, 0
            elif lux_action[2] == 1:
                return ActionType.TRANSFER_ORE, 1

        if lux_action[0] == 2:
            return ActionType.PICKUP_POWER, None

        if lux_action[0] == 5:
            return ActionType.RECHARGE, None

        else:
            return None, None

    rewarded_actions = []
    action_queue_copy = copy.deepcopy(action_queue)
    resource = None
    for idx, lux_action in enumerate(reversed(action_queue_copy)):
        rewarded_action, resource_this_action = lux_action_to_type(lux_action, resource)
        if idx == 0 and rewarded_action is None and lux_action[0] == 0:
            rewarded_action = ActionType.RETURN

        if resource_this_action is not None:
            resource = resource_this_action
        if rewarded_action is not None:
            rewarded_actions.append(rewarded_action)

    return list(reversed(rewarded_actions))

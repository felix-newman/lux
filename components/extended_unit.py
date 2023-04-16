from dataclasses import dataclass
from enum import Enum
from typing import List

from components.actions import ActionSequence, RewardedAction, ActionType
from lux.unit import Unit


class UnitRole(Enum):
    MINER = 1
    DIGGER = 2
    FIGHTER = 3

    def valid_reward_actions(self) -> List[RewardedAction]:
        if self == UnitRole.MINER:
            return [ActionType.MINE_ORE, ActionType.MINE_ICE, ActionType.PICKUP_POWER,
                    ActionType.TRANSFER_ORE, ActionType.TRANSFER_ICE]

        if self == UnitRole.DIGGER:
            return [ActionType.PICKUP_POWER]

        if self == UnitRole.FIGHTER:
            return [ActionType.PICKUP_POWER]


class UnitState(Enum):
    WITH_ICE = 0,
    WITH_ORE = 1,
    WITH_EMPTY_CARGO = 2


@dataclass
class ExtendedUnit:
    unit: Unit
    unit_id: str
    cur_action_sequence: ActionSequence
    role: UnitRole



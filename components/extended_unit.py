from dataclasses import dataclass
from enum import Enum
from typing import List

from components.actions import ActionSequence, RewardedActionType
from lux.unit import Unit


class UnitRole(Enum):
    MINER = 1
    DIGGER = 2
    FIGHTER = 3

    def valid_reward_actions(self) -> List[RewardedActionType]:
        if self == UnitRole.MINER:
            return [RewardedActionType.MINE_ORE, RewardedActionType.MINE_ICE, RewardedActionType.PICKUP_POWER,
                    RewardedActionType.TRANSFER_ORE, RewardedActionType.TRANSFER_ICE]

        if self == UnitRole.DIGGER:
            return [RewardedActionType.PICKUP_POWER]

        if self == UnitRole.FIGHTER:
            return [RewardedActionType.PICKUP_POWER]


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



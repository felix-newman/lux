from typing import Dict

import numpy as np

from components.actions import ActionSequence, RewardedActionSequence, RewardedActionType
from lux.kit import GameState
from lux.unit import Unit


class ActionSequenceCalculator:
    def convert_to_action_sequence(self, unit: Unit, game_state: GameState,
                                   rewarded_action_sequence: RewardedActionSequence) -> ActionSequence:
        pass


class UnitController:
    def find_optimally_rewarded_action_sequence(self, unit, game_state: GameState,
                                                reward_maps: Dict[
                                                    RewardedActionType, np.array]) -> RewardedActionSequence:
        pass

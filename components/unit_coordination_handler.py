from typing import Dict

import numpy as np

from components.actions import ActionSequence
from components.constants import MAP_SIZE, RewardedActionType
from lux.kit import GameState


class UnitReference:
    def __init__(self):
        pass


class RewardActionMap:
    def __init__(self):
        self.reward_map = np.zeros(MAP_SIZE, MAP_SIZE)
        self.taker_to_priority = [[dict() for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]


class UnitCoordinationHandler:
    def __init__(self):
        self.reference: Dict[str, UnitReference] = dict()
        self.occupancy_map = np.zeros(MAP_SIZE, MAP_SIZE)
        self.reward_action_maps = Dict[RewardedActionType, RewardActionMap]

    def grant_rewards(self, unit_id: str, action_sequence: ActionSequence):
        pass

    def calculate_unit_reward_maps(self, game_state: GameState):
        pass

    def clean_up_unit(self, unit_id: str):
        pass

    def get_reward_maps(self):
        pass

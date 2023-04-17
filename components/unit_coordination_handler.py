from dataclasses import dataclass
from typing import Dict

import numpy as np

from components.actions import ActionSequence, RewardedAction, rewarded_actions, ActionType
from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState


@dataclass
class UnitReference:
    """Positive rewards taken for the corresponding action type by this unit"""
    reward_masks: Dict[RewardedAction, np.array]

    def get_reward_positions(self, action_type: RewardedAction) -> np.array:
        return np.argwhere(self.reward_masks[action_type] == 1)


class RewardActionHandler:
    """
    Object to hold all reward related information regarding one RewardedActionType
    - reward_mask: mask that contains the information of whether the reward is available or not. Only integers, a value n > 1
    - taken_rewards: tracks how often a reward has been taken at a given position
    implies that n units in total can take this reward.
    """

    def __init__(self, action_type: RewardedAction):
        self.type: RewardedAction = action_type
        self.reward_map: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.future_discount_factor = np.zeros((MAP_SIZE, MAP_SIZE))
        self.reward_mask: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.taken_rewards: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.taker_to_priority = [[dict() for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]

    def grant_reward(self, unit_id: str, unit_reference: UnitReference, reward: float):
        self.taken_rewards += unit_reference.reward_masks[self.type]
        for x, y in unit_reference.get_reward_positions(self.type):
            self.taker_to_priority[x][y][unit_id] = reward

    def clean_up(self, unit_id: str, unit_reference: UnitReference):
        self.taken_rewards -= unit_reference.reward_masks[self.type]
        for x, y in unit_reference.get_reward_positions(self.type):
            del self.taker_to_priority[x][y][unit_id]

    def self_aware_reward_mask(self, unit_reference: UnitReference) -> np.array:
        if self.type in unit_reference.reward_masks:
            return self.reward_mask - self.taken_rewards + unit_reference.reward_masks[self.type]
        else:
            return self.reward_mask - self.taken_rewards

    @property
    def counterfactual_reward_mask(self) -> np.array:
        return self.reward_mask


class UnitCoordinationHandler:
    def __init__(self, self_player: str):
        self.references: Dict[str, UnitReference] = dict()
        self.occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        self.reward_action_handler: Dict[RewardedAction, RewardActionHandler] = dict()

        self.self_player = self_player

    def grant_rewards(self, unit_id: str, action_sequence: ActionSequence):
        """
        First cleans all references to this unit, then updates the reward maps for the different actions and computes a
        reference to all modifications of this unit in order to be able to roll it back properly
        """

        self.clean_up_unit(unit_id)

        reward_masks = self._build_unit_reward_masks(action_sequence)

        reference = UnitReference(reward_masks)
        for action_type in reference.reward_masks.keys():
            self.reward_action_handler[action_type].grant_reward(unit_id=unit_id, unit_reference=reference,
                                                                 reward=action_sequence.reward)

        self.references[unit_id] = reference

    @staticmethod
    def _build_unit_reward_masks(action_sequence: ActionSequence):
        """
        Go through all actions in action sequence and build a reward_mask for all of them. Items of the same action type
        can occur multiple times, therefore masks have to be updated if they already exist for one type.
        """
        reward_masks = dict()
        for item in action_sequence.action_items:
            if not item.type.is_rewarded_action:
                continue

            mask = np.zeros((MAP_SIZE, MAP_SIZE))
            x, y = item.position
            if item.type.is_factory_action:
                mask[x - 1:x + 2, y - 1: y + 2] = 1
            else:
                mask[x, y] = 1

            rewarded_action_type = item.type
            if rewarded_action_type in reward_masks:
                reward_masks[rewarded_action_type] += mask
            else:
                reward_masks[rewarded_action_type] = mask

        return reward_masks

    def clean_up_unit(self, unit_id: str):
        if unit_id in self.references:
            unit_reference = self.references[unit_id]
            for action_type in unit_reference.reward_masks.keys():
                self.reward_action_handler[action_type].clean_up(unit_id=unit_id, unit_reference=unit_reference)
            del self.references[unit_id]

    def initialize_unit_reward_handler(self, game_state: ExtendedGameState):
        for action_type in rewarded_actions:
            self.reward_action_handler[action_type] = self._build_reward_action_masks(action_type, game_state)

    def get_self_aware_reward_masks(self, unit_id: str):
        """
        Get the reward map with all rewards taken by other units applied and the rewards taken by this unit added
        back. Important if reordering/shortening of current task would be beneficial.
        """
        unit_reference = self.references[unit_id]
        return {k: v.self_aware_reward_mask(unit_reference) for k, v in self.reward_action_handler.items()}

    def get_counterfactual_reward_masks(self):
        """
        Return the reward maps as-is, that is without any rewards taken.
        """
        return {k: v.counterfactual_reward_mask for k, v in self.reward_action_handler.items()}

    def update_reward_handler(self, action_type: RewardedAction, reward_map: np.array,
                              future_discount_factor: np.array):
        raise NotImplemented

    @staticmethod
    def _build_reward_action_masks(action_type: RewardedAction,
                                   game_state: ExtendedGameState) -> RewardActionHandler:
        if action_type is ActionType.MINE_ICE:
            reward_action_mask = RewardActionHandler(action_type)
            reward_action_mask.reward_mask = game_state.board.ice
            reward_action_mask.reward_map = game_state.board.ice
            return reward_action_mask

        elif action_type is ActionType.MINE_ORE:
            reward_action_mask = RewardActionHandler(action_type)
            reward_action_mask.reward_mask = game_state.board.ore
            reward_action_mask.reward_map = game_state.board.ore
            return reward_action_mask
        elif action_type is ActionType.TRANSFER_ICE:
            reward_action_mask = RewardActionHandler(action_type)
            reward_action_mask.reward_mask = game_state.player_factories * 10
            reward_action_mask.reward_map = game_state.player_factories * 10
            return reward_action_mask
        elif action_type is ActionType.TRANSFER_ORE:
            reward_action_mask = RewardActionHandler(action_type)
            reward_action_mask.reward_mask = game_state.player_factories * 10
            reward_action_mask.reward_map = game_state.player_factories * 10

            return reward_action_mask
        elif action_type is ActionType.PICKUP_POWER:
            reward_action_mask = RewardActionHandler(action_type)
            reward_action_mask.reward_mask = game_state.player_factories * 10
            reward_action_mask.reward_map = game_state.player_factories * 0.0001

            return reward_action_mask

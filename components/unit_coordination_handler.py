import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from components.actions import ActionSequence, RewardedAction, rewarded_actions, ActionType
from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState
from components.utils import get_position_after_lux_action
from lux.factory import Factory


@dataclass
class UnitReference:
    """Positive rewards taken for the corresponding action type by this unit"""
    reward_masks: Dict[RewardedAction, np.array]
    reward_maps: Dict[RewardedAction, np.array]

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
        self._taken_rewards_map: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self._reward_mask: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self._taken_rewards_mask: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.taker_to_priority = [[dict() for j in range(MAP_SIZE)] for i in range(MAP_SIZE)]

    def grant_reward(self, unit_id: str, unit_reference: UnitReference, reward: float):
        self._taken_rewards_mask += unit_reference.reward_masks[self.type]
        self._taken_rewards_map += unit_reference.reward_maps[self.type]

        for x, y in unit_reference.get_reward_positions(self.type):
            self.taker_to_priority[x][y][unit_id] = reward

    def clean_up(self, unit_id: str, unit_reference: UnitReference):
        self._taken_rewards_mask -= unit_reference.reward_masks[self.type]
        self._taken_rewards_map -= unit_reference.reward_maps[self.type]
        for x, y in unit_reference.get_reward_positions(self.type):
            del self.taker_to_priority[x][y][unit_id]

    @property
    def actual_reward_mask(self) -> np.array:
        return self._reward_mask - self._taken_rewards_mask

    @property
    def actual_reward_map(self) -> np.array:
        return self.reward_map - self._taken_rewards_map

    def self_aware_reward_mask(self, unit_reference: UnitReference) -> np.array:
        if self.type in unit_reference.reward_masks:
            return self._reward_mask - self._taken_rewards_mask + unit_reference.reward_masks[self.type]
        else:
            return self._reward_mask - self._taken_rewards_mask

    def self_aware_reward_map(self, unit_reference: UnitReference) -> np.array:
        if self.type in unit_reference.reward_maps:
            return self.actual_reward_map - self._taken_rewards_map + unit_reference.reward_maps[self.type]
        else:
            return self.actual_reward_map - self._taken_rewards_map


class UnitCoordinationHandler:
    def __init__(self, self_player: str):
        self.references: Dict[str, UnitReference] = dict()
        self.occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        self.reward_action_handler: Dict[RewardedAction, RewardActionHandler] = dict()

        self.self_player = self_player

    def build_occupancy_map(self, game_state: ExtendedGameState, opponent_player: str):
        self.occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        for _, factory in game_state.game_state.factories[opponent_player].items():
            pos_slice = factory.pos_slice
            self.occupancy_map[pos_slice] = 1

    def mark_field_as_occupied(self, x: int, y: int, unit_id: str):
        if 0 < x < MAP_SIZE and 0 < y < MAP_SIZE:
            unit_number = int(unit_id.split("_")[1])
            self.occupancy_map[x, y] = unit_number

    def check_field_occupied(self, x: int, y: int, unit_id: str) -> bool:
        if 0 < x < MAP_SIZE and 0 < y < MAP_SIZE:
            unit_number = int(unit_id.split("_")[1])
            return self.occupancy_map[x, y] != unit_number and self.occupancy_map[x, y] != 0
        else:
            return True

    def collision_after_lux_action(self, lux_action: np.array, cur_pos: np.array, unit_id: str) -> bool:
        pos = get_position_after_lux_action(lux_action, cur_pos)
        return self.check_field_occupied(pos[0], pos[1], unit_id)

    def register_lux_action_for_collision(self, lux_action: np.array, cur_pos: np.array, unit_id: str):
        pos = get_position_after_lux_action(lux_action, cur_pos)
        self.mark_field_as_occupied(pos[0], pos[1], unit_id)

    def grant_rewards(self, unit_id: str, action_sequence: ActionSequence, game_state: ExtendedGameState):
        """
        First cleans all references to this unit, then updates the reward maps for the different actions and computes a
        reference to all modifications of this unit in order to be able to roll it back properly
        """

        self.clean_up_unit(unit_id)

        reward_masks, reward_maps = self._build_unit_reward_masks(action_sequence, game_state)

        reference = UnitReference(reward_masks=reward_masks, reward_maps=reward_maps)
        for action_type in reference.reward_masks.keys():
            self.reward_action_handler[action_type].grant_reward(unit_id=unit_id, unit_reference=reference,
                                                                 reward=action_sequence.reward)

        self.references[unit_id] = reference

    @staticmethod
    def _build_unit_reward_masks(action_sequence: ActionSequence, game_state: ExtendedGameState) -> Tuple[Dict[RewardedAction, np.array], Dict[RewardedAction, np.array]]:
        """
        Go through all actions in action sequence and build a reward_mask for all of them. Items of the same action type
        can occur multiple times, therefore masks have to be updated if they already exist for one type.
        """
        reward_masks = dict()
        reward_maps = dict()
        for item in action_sequence.action_items:
            if not item.type.is_rewarded_action:
                continue

            mask = np.zeros((MAP_SIZE, MAP_SIZE))
            reward_map = np.zeros((MAP_SIZE, MAP_SIZE))
            if item.type.is_factory_action:
                factory_slice = game_state.get_factory_slice_at_position(item.position)
                if factory_slice is None:
                    print("No factory at position", item.position, file = sys.stderr)
                    continue
                mask[factory_slice] = 1
                reward_map[factory_slice] = item.reward
            else:
                x, y = item.position
                mask[x, y] = 1
                reward_map[x, y] = item.reward

            rewarded_action_type = item.type
            if rewarded_action_type in reward_masks:
                reward_masks[rewarded_action_type] += mask
                reward_maps[rewarded_action_type] += reward_map
            else:
                reward_masks[rewarded_action_type] = mask
                reward_maps[rewarded_action_type] = reward_map

        return reward_masks, reward_maps

    def clean_up_unit(self, unit_id: str):
        if unit_id in self.references:
            unit_reference = self.references[unit_id]
            for action_type in unit_reference.reward_masks.keys():
                self.reward_action_handler[action_type].clean_up(unit_id=unit_id, unit_reference=unit_reference)
            del self.references[unit_id]

    def clean_up_action_type(self, action_type: ActionType, unit_id: str):
        if unit_id in self.references:
            unit_reference = self.references[unit_id]
            if action_type in unit_reference.reward_masks:
                self.reward_action_handler[action_type].clean_up(unit_id=unit_id, unit_reference=unit_reference)
                del unit_reference.reward_masks[action_type]
                del unit_reference.reward_maps[action_type]

    def initialize_unit_reward_handler(self, game_state: ExtendedGameState):
        for action_type in rewarded_actions:
            self.reward_action_handler[action_type] = self._build_reward_action_handler(action_type, game_state)

    def get_reward_mask(self, action_type: ActionType) -> np.array:
        return self.reward_action_handler[action_type].actual_reward_mask

    def get_reward_map(self, action_type: ActionType) -> np.array:

        return self.reward_action_handler[action_type].reward_map

    def get_actual_reward_mask(self, action_type: ActionType) -> np.array:
        return self.reward_action_handler[action_type].actual_reward_mask

    def get_actual_reward_map(self, action_type: ActionType) -> np.array:

        return self.reward_action_handler[action_type].actual_reward_map

    def update_reward_handler(self, action_type: RewardedAction, reward_map: np.array, reward_mask: np.array):
        self.reward_action_handler[action_type].reward_map = reward_map
        self.reward_action_handler[action_type]._reward_mask = reward_mask

    def update_factory_rewards(self, action_type: RewardedAction, value: float, factory: Factory):
        self.reward_action_handler[action_type].reward_map[factory.pos_slice] = value

    @staticmethod
    def _build_reward_action_handler(action_type: RewardedAction,
                                     game_state: ExtendedGameState) -> RewardActionHandler:
        if action_type is ActionType.MINE_ICE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.board.ice
            reward_action_handler.reward_map = game_state.board.ice
            return reward_action_handler

        elif action_type is ActionType.MINE_ORE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.board.ore
            reward_action_handler.reward_map = game_state.board.ore
            return reward_action_handler
        elif action_type is ActionType.TRANSFER_ICE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.player_factories * 2
            reward_action_handler.reward_map = game_state.player_factories * 10
            return reward_action_handler
        elif action_type is ActionType.TRANSFER_ORE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.player_factories * 2
            reward_action_handler.reward_map = game_state.player_factories * 10

            return reward_action_handler
        elif action_type is ActionType.PICKUP_POWER:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.player_factories * 1000
            reward_action_handler.reward_map = game_state.player_factories
            return reward_action_handler
        elif action_type is ActionType.DIG:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = np.ones(
                (MAP_SIZE, MAP_SIZE)) - game_state.board.ice - game_state.board.ore - np.where(game_state.board.factory_occupancy_map >= 0, 1, 0)
            reward_action_handler.reward_map = (np.ones(
                (MAP_SIZE, MAP_SIZE)) - game_state.board.ice - game_state.board.ore - np.where(game_state.board.factory_occupancy_map >= 0,
                                                                                               1, 0)) * game_state.board.rubble
            return reward_action_handler
        elif action_type is ActionType.RETURN:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.player_factories * 1000
            reward_action_handler.reward_map = game_state.player_factories
            return reward_action_handler

        elif action_type is ActionType.RECHARGE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = np.ones((MAP_SIZE, MAP_SIZE))
            reward_action_handler.reward_map = np.ones((MAP_SIZE, MAP_SIZE))
            return reward_action_handler

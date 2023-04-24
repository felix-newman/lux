import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from components.actions import ActionSequence, RewardedAction, rewarded_actions, ActionType
from components.constants import MAP_SIZE
from components.enemy_map import EnemyMap
from components.extended_game_state import ExtendedGameState
from components.utils import get_position_after_lux_action
from lux.factory import Factory
from lux.unit import Unit


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
    def __init__(self, self_player: str, opp_player: str):
        self.references: Dict[str, UnitReference] = dict()
        self._occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        self.reward_action_handler: Dict[RewardedAction, RewardActionHandler] = dict()
        self.enemy_map: EnemyMap = EnemyMap(opp_player=opp_player)

        self.self_player = self_player
        self.opp_player = opp_player
        self.own_lichen_strains: List[int] = []
        self.enemy_lichen_strains: List[int] = []

    def build_occupancy_map(self, game_state: ExtendedGameState, opponent_player: str):
        self._occupancy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        for _, factory in game_state.game_state.factories[opponent_player].items():
            pos_slice = factory.pos_slice
            self._occupancy_map[pos_slice] = 1

    def mark_field_as_occupied(self, x: int, y: int, unit_id: str):
        if 0 < x < MAP_SIZE and 0 < y < MAP_SIZE:
            unit_number = int(unit_id.split("_")[1])
            self._occupancy_map[x, y] = unit_number

    def update_enemy_map(self, game_state: ExtendedGameState):
        self.enemy_map.update(game_state)
        self.reward_action_handler[ActionType.FIGHT]._reward_mask = self.enemy_map.enemy_map

    def fight_possible_after_lux_action(self, lux_action: np.array, cur_pos: np.array) -> bool:
        pos = get_position_after_lux_action(lux_action, cur_pos)
        return self.enemy_map.is_fighting_position(pos[0], pos[1])

    def update_loot_map(self, game_state: ExtendedGameState):
        reward_mask = np.zeros((MAP_SIZE, MAP_SIZE))
        for strain_id in self.enemy_lichen_strains:
            reward_mask += np.where(game_state.board.lichen_strains == strain_id, 1, 0)

        self.reward_action_handler[ActionType.LOOT]._reward_mask = reward_mask

    def on_fight_field(self, cur_pos: np.array):
        return self.enemy_map.is_fighting_position(cur_pos[0], cur_pos[1])

    def get_enemy_adjusted_occupancy_map(self, unit: Unit):
        return self._occupancy_map + self.enemy_map.superior_enemies_map(unit)

    def get_strongest_enemy(self, pos: np.array) -> Tuple[int, int]:
        return self.enemy_map.get_strongest_enemy(pos)

    def check_field_occupied(self, x: int, y: int, unit: Unit) -> bool:
        if 0 < x < MAP_SIZE and 0 < y < MAP_SIZE:
            unit_number = int(unit.unit_id.split("_")[1])
            occupancy_map = self.get_enemy_adjusted_occupancy_map(unit)
            return occupancy_map[x, y] != unit_number and occupancy_map[x, y] != 0
        else:
            return True

    def collision_after_lux_action(self, lux_action: np.array, cur_pos: np.array, unit: Unit) -> bool:
        pos = get_position_after_lux_action(lux_action, cur_pos)
        return self.check_field_occupied(pos[0], pos[1], unit)

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
    def _build_unit_reward_masks(action_sequence: ActionSequence, game_state: ExtendedGameState) -> Tuple[
        Dict[RewardedAction, np.array], Dict[RewardedAction, np.array]]:
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
                    print("No factory at position", item.position, file=sys.stderr)
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
        for factory_id, factory in game_state.game_state.factories[self.self_player].items():
            self.own_lichen_strains.append(factory.strain_id)

        for factory_id, factory in game_state.game_state.factories[self.opp_player].items():
            self.enemy_lichen_strains.append(factory.strain_id)

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

    def update_factory_rewards(self, action_type: RewardedAction, factory: Factory, reward_value: float, mask_value: float = 9999.0):
        self.reward_action_handler[action_type].reward_map[factory.pos_slice] = reward_value
        self.reward_action_handler[action_type]._reward_mask[factory.pos_slice] = mask_value

    def _build_reward_action_handler(self, action_type: RewardedAction,
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
            reward_action_handler._reward_mask = game_state.player_factories * 1
            reward_action_handler.reward_map = game_state.player_factories * 10
            return reward_action_handler
        elif action_type is ActionType.TRANSFER_ORE:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = game_state.player_factories * 3
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
                (MAP_SIZE, MAP_SIZE)) - game_state.board.ice - game_state.board.ore - np.where(game_state.board.factory_occupancy_map >= 0,
                                                                                               1, 0)
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

        elif action_type is ActionType.FIGHT:
            reward_action_handler = RewardActionHandler(action_type)
            reward_action_handler._reward_mask = self.enemy_map.enemy_map
            reward_action_handler.reward_map = np.ones((MAP_SIZE, MAP_SIZE))
            return reward_action_handler

        elif action_type is ActionType.LOOT:
            reward_action_handler = RewardActionHandler(action_type)

            reward_mask = np.zeros((MAP_SIZE, MAP_SIZE))
            for strain_id in self.enemy_lichen_strains:
                reward_mask += np.where(game_state.board.lichen_strains == strain_id, 1, 0)

            reward_action_handler._reward_mask = reward_mask
            reward_action_handler.reward_map = np.ones((MAP_SIZE, MAP_SIZE)) * 2
            return reward_action_handler

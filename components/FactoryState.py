import random
import sys

from enum import Enum
from typing import Dict, List, Union

import numpy as np

from components.actions import ActionType
from components.extended_game_state import ExtendedGameState
from components.extended_unit import UnitMetadata, UnitRole
from components.unit_coordination_handler import UnitCoordinationHandler
from lux.factory import Factory


class FactoryAction(Enum):
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2
    NOOP = 3


class FactoryState:

    def __init__(self):
        self.factory: Union[None, Factory] = None
        self.ore_income_per_round: float = 0.0
        self.ice_income_per_round: float = 0.0

        self.ore_reward: float = 5.0
        self.ice_reward: float = 1.0
        self.available_power: float = 0.0
        self.power_bank = 0.0
        self.savings_for_build = 10.0

        self.max_ice_miners = 1.0
        self.max_ore_miners = 3.0

        self.ore_miners_to_type: Dict[str, str] = dict()
        self.ice_miners_to_type: Dict[str, str] = dict()

        self.diggers_to_type: Dict[str, str] = dict()
        self.fighters_to_type: Dict[str, str] = dict()

        self.next_build_action: FactoryAction = FactoryAction.BUILD_HEAVY
        self.next_role: UnitRole = UnitRole.MINER
        self.next_action: FactoryAction = FactoryAction.BUILD_HEAVY

        self.recalculate_next_build_and_role_in = 1
        self.last_actions: List[FactoryAction] = []

        self.built_miners_to_type: Dict[str, str] = dict()
        self.built_diggers_to_type: Dict[str, str] = dict()
        self.built_fighters_to_type: Dict[str, str] = dict()

    def update_stats(self, factory: Factory, unit_coordination_handler: UnitCoordinationHandler, game_state: ExtendedGameState,
                     self_player: str, tracked_units: Dict[str, UnitMetadata]):

        self.factory = factory
        self.recalculate_next_build_and_role_in -= 1
        self.power_bank += self.savings_for_build

        x, y = factory.pos
        ice_miners = unit_coordination_handler.reward_action_handler[ActionType.TRANSFER_ICE].taker_to_priority[x][y].keys()
        for miner_id in ice_miners:
            unit = game_state.game_state.units[self_player][miner_id]
            self.ice_miners_to_type[miner_id] = unit.unit_type

            digging_speed = 20 if unit.unit_type == "HEAVY" else 2
            self.ice_income_per_round += digging_speed * self.n_mining_steps_in(unit.action_queue)

        ore_miners = unit_coordination_handler.reward_action_handler[ActionType.TRANSFER_ORE].taker_to_priority[x][y].keys()
        for miner_id in ore_miners:
            unit = game_state.game_state.units[self_player][miner_id]
            self.ore_miners_to_type[miner_id] = unit.unit_type

            digging_speed = 20 if unit.unit_type == "HEAVY" else 2
            self.ore_income_per_round += digging_speed * self.n_mining_steps_in(unit.action_queue)

        self.ore_reward = unit_coordination_handler.reward_action_handler[ActionType.TRANSFER_ORE].reward_map[x][y]
        self.ice_reward = unit_coordination_handler.reward_action_handler[ActionType.TRANSFER_ICE].reward_map[x][y]


        miners_to_remove = []
        for unit_id in self.built_miners_to_type:
            if unit_id not in tracked_units:
                miners_to_remove.append(unit_id)
        for unit_id in miners_to_remove:
            del self.built_miners_to_type[unit_id]

        diggers_to_remove = []
        for unit_id in self.built_diggers_to_type:
            if unit_id not in tracked_units:
                diggers_to_remove.append(unit_id)
        for unit_id in diggers_to_remove:
            del self.built_diggers_to_type[unit_id]

        fighters_to_remove = []
        for unit_id in self.built_fighters_to_type:
            if unit_id not in tracked_units:
                fighters_to_remove.append(unit_id)

        for unit_id in fighters_to_remove:
            del self.built_fighters_to_type[unit_id]

    def get_next_action(self) -> np.array:
        if self.next_action == FactoryAction.BUILD_LIGHT:
            return self.factory.build_light()
        elif self.next_action == FactoryAction.BUILD_HEAVY:
            return self.factory.build_heavy()
        elif self.next_action == FactoryAction.WATER:
            return self.factory.water()
        else:
            return None

    def calculate_next_rewards(self, game_state: ExtendedGameState):
        self.available_power = max(0.0, self.factory.power - self.power_bank)

        if game_state.real_env_steps < 500:
            self.ore_reward = 5.0
            self.ice_reward = 1.0
        else:
            self.ore_reward = 1.0
            self.ice_reward = 1.0

        if self.factory.cargo.water < 100 and game_state.real_env_steps < 900:
            self.ore_reward = 1.0
            self.ice_reward = 1.0
        elif self.factory.cargo.water < 75 and game_state.real_env_steps < 900:
            self.ore_reward = 1.0
            self.ice_reward = 5.0
            # TODO do role switch here


    # TODO ensure that heavy robots take most important task
    def calculate_next_build_and_role(self, game_state: ExtendedGameState):
        light_ice_miners, heavy_ice_miners = self.count_types(self.ice_miners_to_type)
        light_ore_miners, heavy_ore_miners = self.count_types(self.ore_miners_to_type)
        heavy_miners = heavy_ice_miners + heavy_ore_miners
        light_miners = light_ice_miners + light_ore_miners

        if game_state.real_env_steps == 0:
            self.next_build_action = FactoryAction.BUILD_HEAVY
            self.next_role = UnitRole.MINER
            return

        if self.ore_income_per_round >= 4:
            if heavy_miners < 2:
                self.next_build_action = FactoryAction.BUILD_HEAVY
                self.next_role = UnitRole.MINER
            else:
                # TODO take rubble into account and closeness to enemy factory
                if game_state.real_env_steps < 500:
                    self.next_build_action = random.choices([FactoryAction.BUILD_LIGHT, FactoryAction.BUILD_HEAVY], weights=[0.7, 0.3])[0]
                    probabilities = [0.5, 0.5] if self.next_build_action == FactoryAction.BUILD_HEAVY else [0.3, 0.7]
                    self.next_role = random.choices([UnitRole.DIGGER, UnitRole.FIGHTER], weights=probabilities)[0]

                # TODO take rubble into account and enemy bots
                else:
                    self.next_build_action = random.choices([FactoryAction.BUILD_LIGHT, FactoryAction.BUILD_HEAVY], weights=[0.9, 0.1])[0]
                    probabilities = [0.5, 0.5] if self.next_build_action == FactoryAction.BUILD_HEAVY else [0.0, 1.0]
                    self.next_role = random.choices([UnitRole.DIGGER, UnitRole.FIGHTER], weights=probabilities)[0]

        elif 0 < self.ore_income_per_round < 4:
            if light_miners < 3:
                self.next_build_action = FactoryAction.BUILD_LIGHT
                self.next_role = UnitRole.MINER
            else:
                self.next_build_action = FactoryAction.BUILD_LIGHT
                probabilities = [0.5, 0.5] if game_state.real_env_steps < 700 else [0.0, 1.0]
                self.next_role = random.choices([UnitRole.DIGGER, UnitRole.FIGHTER], weights=probabilities)[0]

        else:  # no ore income
            if light_miners < 1:  # try with one light miner
                self.next_build_action = FactoryAction.BUILD_LIGHT
                self.next_role = UnitRole.MINER
            else:
                self.next_build_action = FactoryAction.BUILD_LIGHT
                self.next_role = UnitRole.DIGGER

        print(f"Next builds and role wille be: {self.next_build_action}, {self.next_role}", file=sys.stderr)

    def calculate_next_action(self, game_state: ExtendedGameState):
        if self.next_build_action == FactoryAction.BUILD_LIGHT and self.factory.can_build_light(game_state.game_state):
            self.next_action = FactoryAction.BUILD_LIGHT
            return
        elif self.next_build_action == FactoryAction.BUILD_HEAVY and self.factory.can_build_heavy(game_state.game_state):
            self.next_action = FactoryAction.BUILD_HEAVY
            return

        # TODO improve to include alternating watering
        elif game_state.real_env_steps > 800 and self.factory.can_water(game_state) and self.factory.cargo.water - self.factory.water_cost(
                game_state) > 1000 - game_state.real_env_steps:
            self.next_action = FactoryAction.WATER
            return

        self.next_action = FactoryAction.NOOP

    def register_unit_at_factory(self, unit_meta: UnitMetadata):
        if unit_meta.role == UnitRole.MINER:
            self.built_miners_to_type[unit_meta.unit_id] = unit_meta.unit_type
        elif unit_meta.role == UnitRole.DIGGER:
            self.built_diggers_to_type[unit_meta.unit_id] = unit_meta.unit_type
        elif unit_meta.role == UnitRole.FIGHTER:
            self.built_fighters_to_type[unit_meta.unit_id] = unit_meta.unit_type

    @staticmethod
    def count_types(id_to_type: Dict[str, str]):
        light = 0
        heavy = 0
        for unit_type in id_to_type.values():
            if unit_type == "LIGHT":
                light += 1
            else:
                heavy += 1
        return light, heavy

    @staticmethod
    def n_mining_steps_in(lux_action_queue: List[np.array]):
        for action in lux_action_queue:
            if action[0] == 3:
                return action[5]
        return 0

    def register_next_action(self, unit_coordination_handler: UnitCoordinationHandler):
        if self.next_action == FactoryAction.BUILD_LIGHT or self.next_action == FactoryAction.BUILD_HEAVY:
            self.available_power = self.available_power - 500 if self.next_action == FactoryAction.BUILD_HEAVY else self.available_power - 50
            self.power_bank = 0.0

            unit_coordination_handler.mark_field_as_occupied(self.factory.pos[0], self.factory.pos[1], 'unit_99999')
            self.recalculate_next_build_and_role_in = 2

import random
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from components.FactoryState import FactoryState
from components.actions import ActionSequence, ActionType
from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState
from components.extended_unit import UnitMetadata
from components.factory_placement import compute_factory_value_map
from components.unit_controller import UnitController
from components.unit_coordination_handler import UnitCoordinationHandler
from components.utils import get_cheapest_path
from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        random.seed(1)
        self.env_cfg: EnvConfig = env_cfg

        self.factory_value_map = None

        self.unit_controller = UnitController()
        self.unit_coordination_handler = UnitCoordinationHandler(self_player=self.player, opp_player=self.opp_player)

        self.tracked_units: Dict[str, UnitMetadata] = dict()
        self.factory_states: Dict[str, FactoryState] = dict()

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period
            if self.factory_value_map is None:
                self.factory_value_map = compute_factory_value_map(game_state)

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                best_spawn_idx = np.argmax(self.factory_value_map * game_state.board.valid_spawns_mask)
                spawn_loc = np.unravel_index(best_spawn_idx, self.factory_value_map.shape)
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = ExtendedGameState(obs_to_game_state(step, self.env_cfg, obs), player=self.player)
        if game_state.real_env_steps == 0:
            self.setup(game_state)

        self.unit_coordination_handler.build_occupancy_map(game_state, self.opp_player)
        self.unit_coordination_handler.update_enemy_map(game_state)
        self.unit_coordination_handler.update_loot_map(game_state)

        new_dig_reward_mask, new_dig_reward_map = self.calculate_next_dig_mask(game_state)

        self.unit_coordination_handler.update_reward_handler(ActionType.DIG, new_dig_reward_map, new_dig_reward_mask)

        # clean up dead units, units with empty action sequences
        units_to_remove = []
        for unit_id in self.tracked_units:
            unit = game_state.game_state.units[self.player].get(unit_id)
            if unit is None:
                units_to_remove.append(unit_id)
                self.unit_coordination_handler.clean_up_unit(unit_id)

        for unit_id in units_to_remove:
            del self.tracked_units[unit_id]

        for unit_id, unit_meta in self.tracked_units.items():
            if unit_meta.last_action == ActionType.PICKUP_POWER:
                self.unit_coordination_handler.clean_up_action_type(unit_id=unit_id, action_type=ActionType.PICKUP_POWER)

        for factory_id, factory in game_state.game_state.factories[self.player].items():
            factory_state = self.factory_states[factory_id]

            factory_state.update_stats(factory=factory, unit_coordination_handler=self.unit_coordination_handler,
                                       game_state=game_state, self_player=self.player, tracked_units=self.tracked_units)
            factory_state.calculate_next_rewards(game_state)
            if factory_state.recalculate_next_build_and_role_in == 0:
                factory_state.calculate_next_build_and_role(game_state)
            factory_state.calculate_next_action(game_state)

            factory_state.register_next_action(unit_coordination_handler=self.unit_coordination_handler)
            next_action = factory_state.get_next_action()
            if next_action is not None:
                actions[factory_id] = next_action

            self.unit_coordination_handler.update_factory_rewards(ActionType.PICKUP_POWER,
                                                                  reward_value=factory_state.available_power,
                                                                  factory=factory_state.factory)
            self.unit_coordination_handler.update_factory_rewards(ActionType.TRANSFER_ICE, reward_value=factory_state.ice_reward,
                                                                  mask_value=factory_state.max_ice_miners,
                                                                  factory=factory_state.factory)
            self.unit_coordination_handler.update_factory_rewards(ActionType.TRANSFER_ORE, reward_value=factory_state.ore_reward,
                                                                  mask_value=factory_state.max_ore_miners,
                                                                  factory=factory_state.factory)

        # assign tasks to units
        for unit_id, unit in game_state.game_state.units[self.player].items():
            if unit_id not in self.tracked_units:
                closest_factory = self._find_closest_factory(unit.pos, game_state)
                unit_role = self.factory_states[closest_factory].next_role
                factory_mask = np.zeros((MAP_SIZE, MAP_SIZE))
                factory_mask[game_state.game_state.factories[self.player][closest_factory].pos_slice] = 1
                unit_meta = UnitMetadata(unit_id=unit_id, role=unit_role, unit_type=unit.unit_type,
                                         cur_action_sequence=ActionSequence(action_items=[], reward=0, remaining_rewards=[]),
                                         last_action=None, factory_id=closest_factory,
                                         factory_mask=factory_mask)
                self.tracked_units[unit_id] = unit_meta
                self.factory_states[closest_factory].register_unit_at_factory(unit_meta)

        sorted_units = sorted(self.tracked_units.items(),
                              key=lambda x: (1, -game_state.game_state.units[self.player][x[0]].power) if x[1].unit_type == 'HEAVY' else (
                                  0, -game_state.game_state.units[self.player][x[0]].power), reverse=True)

        for unit_id, unit_meta in sorted_units:
            unit = game_state.game_state.units[self.player][unit_id]
            action_sequence, role_change_requested = self.unit_controller.update_action_queue(unit=unit, unit_meta=unit_meta,
                                                                                              unit_coordination_handler=self.unit_coordination_handler,
                                                                                              game_state=game_state)

            if not action_sequence.empty:
                self.unit_coordination_handler.grant_rewards(unit_id=unit_id, action_sequence=action_sequence, game_state=game_state)
                unit_meta.cur_action_sequence = action_sequence

                lux_action_queue = unit_meta.cur_action_sequence.to_lux_action_queue()
                actions[unit_id] = lux_action_queue
                # print(unit.pos, unit_id, unit_meta.cur_action_sequence, file=sys.stderr)

                next_action = lux_action_queue[0] if len(lux_action_queue) > 0 else None
                if next_action is not None:
                    self.unit_coordination_handler.register_lux_action_for_collision(next_action, unit.pos, unit_id)

            else:
                # print(f"No action queue update for {unit_id}", file=sys.stderr)
                if len(unit.action_queue) > 0:
                    # TODO only do if robot has enough power to execute action
                    next_action = unit.action_queue[0]
                    self.unit_coordination_handler.register_lux_action_for_collision(next_action, unit.pos, unit_id)
                else:
                    self.unit_coordination_handler.mark_field_as_occupied(unit.pos[0], unit.pos[1], unit_id)

        print(f"Step: {step}", file=sys.stderr)

        for unit_id, unit in game_state.game_state.units[self.player].items():
            if len(unit.action_queue) > 0:
                next_action = unit.action_queue[0]
                if next_action[0] == 2:  # PICKUP_POWER
                    self.tracked_units[unit_id].last_action = ActionType.PICKUP_POWER
                else:
                    self.tracked_units[unit_id].last_action = None

        return actions

    def setup(self, game_state: ExtendedGameState):
        self.unit_coordination_handler.initialize_unit_reward_handler(game_state)
        for factory_id, factory in game_state.game_state.factories[self.player].items():
            self.factory_states[factory_id] = FactoryState()

    def _find_closest_factory(self, pos: np.array, game_state: ExtendedGameState) -> str:
        closest_distance = np.inf
        closest_factory_id = None

        for factory_id, factory in game_state.game_state.factories[self.player].items():
            distance = np.sum(np.abs(factory.pos - pos))
            if distance < closest_distance:
                closest_distance = distance
                closest_factory_id = factory_id

        return closest_factory_id

    def calculate_next_dig_mask(self, game_state: ExtendedGameState):
        new_dig_reward_map = np.ones((MAP_SIZE, MAP_SIZE))*5
        inv_rubble = 100 - game_state.board.rubble
        dig_needed = np.where(inv_rubble < 100, inv_rubble, 0)
        easy_dig = np.where(dig_needed > 50, 1, 0)
        new_dig_reward_mask = np.zeros((MAP_SIZE, MAP_SIZE))
        for factory_id, factory in game_state.game_state.factories[self.player].items():
            pos = factory.pos
            distance_map = np.sum(np.abs(np.indices((48, 48)) - np.array(pos)[:, None, None]), axis=0)
            close_points = np.where(distance_map <= 6, 1, 0)

            target_point_mask = np.logical_and(close_points, easy_dig)
            if np.max(target_point_mask) == 0:
                new_dig_reward_mask += close_points
            else:
                connected_points = np.argwhere(target_point_mask == 1)[:8]

                for point in connected_points:
                    points =np.array(get_cheapest_path(pos, point, game_state.board.rubble))
                    new_dig_reward_mask[points[:, 0], points[:, 1]] = 1

        new_dig_reward_mask = np.where(game_state.board.factory_occupancy_map >= 0, 0, new_dig_reward_mask)

        new_dig_reward_mask += easy_dig
        new_dig_reward_mask = np.clip(new_dig_reward_mask, 0, 1)
        new_dig_reward_mask = np.where(game_state.board.rubble > 0, new_dig_reward_mask, 0)
        if game_state.real_env_steps % 10 == 0 and self.player == "player_0":
            # save dig reward map every 20 steps
            plt.imsave(f"videos/dig_reward_mask_{game_state.real_env_steps}.png", new_dig_reward_mask.T)

        return new_dig_reward_mask, new_dig_reward_map


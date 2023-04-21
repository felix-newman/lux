import sys
from typing import Dict, List

import numpy as np

from components.actions import ActionSequence, ActionType
from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState
from components.extended_unit import UnitRole, UnitMetadata
from components.factory_placement import compute_factory_value_map
from components.unit_controller import UnitController
from components.unit_coordination_handler import UnitCoordinationHandler
from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from lux.utils import my_turn_to_place_factory


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        self.factory_value_map = None

        self.unit_controller = UnitController()
        self.unit_coordination_handler = UnitCoordinationHandler(self_player=self.player)

        self.tracked_units: Dict[str, UnitMetadata] = dict()
        self.role_switches: List[str] = list()

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

        # TODO factory logic and updating of coordination handler
        for factory_id, factory in game_state.game_state.factories[self.player].items():
            if factory.can_build_heavy(game_state):
                actions[factory_id] = factory.build_heavy()
                self.unit_coordination_handler.mark_field_as_occupied(factory.pos[0], factory.pos[1], 'unit_99999')

            elif factory.can_build_light(game_state):
                actions[factory_id] = factory.build_light()
                self.unit_coordination_handler.mark_field_as_occupied(factory.pos[0], factory.pos[1], 'unit_99999')

            elif factory.can_water(game_state) and factory.cargo.water > 20:
                actions[factory_id] = factory.water()

        # assign tasks to units
        for unit_id, unit in game_state.game_state.units[self.player].items():
            if unit_id not in self.tracked_units:
                self.tracked_units[unit_id] = UnitMetadata(unit_id=unit_id, role=UnitRole.MINER, unit_type=unit.unit_type,
                                                           cur_action_sequence=ActionSequence(action_items=[], reward=0,
                                                                                              remaining_rewards=[]), last_action=None)

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

        for unit_id in self.role_switches:
            unit = self.tracked_units.get(unit_id)
            if unit is not None:
                if self.tracked_units[unit_id].role == UnitRole.MINER:
                    self.tracked_units[unit_id].role = UnitRole.DIGGER
                # elif self.tracked_units[unit_id].role == UnitRole.DIGGER:
                #     self.tracked_units[unit_id].role = UnitRole.MINER

        for factory_id, factory in game_state.game_state.factories[self.player].items():
            self.unit_coordination_handler.update_factory_rewards(ActionType.PICKUP_POWER, value=factory.power, factory=factory)

        new_dig_reward_map = (np.ones(
                (MAP_SIZE, MAP_SIZE)) - game_state.board.ice - game_state.board.ore - np.where(game_state.board.factory_occupancy_map >= 0,
                                                                                               1, 0)) * game_state.board.rubble
        new_dig_reward_mask = np.where(new_dig_reward_map > 0, 1, 0)
        self.unit_coordination_handler.update_reward_handler(ActionType.DIG, new_dig_reward_map, new_dig_reward_mask)

        # update unit action sequences
        sorted_units = sorted(self.tracked_units.items(), key=lambda x: 1 if x[1].unit_type == 'HEAVY' else 0, reverse=True)

        for unit_id, unit_meta in sorted_units:
            unit = game_state.game_state.units[self.player][unit_id]
            action_sequence, role_change_requested = self.unit_controller.update_action_queue(unit=unit, unit_meta=unit_meta,
                                                                                              unit_coordination_handler=self.unit_coordination_handler,
                                                                                              game_state=game_state)

            if not action_sequence.empty:
                self.unit_coordination_handler.grant_rewards(unit_id=unit_id, action_sequence=action_sequence)
                unit_meta.cur_action_sequence = action_sequence

                lux_action_queue = unit_meta.cur_action_sequence.to_lux_action_queue()
                actions[unit_id] = lux_action_queue
                print(unit.pos, unit_id, unit_meta.cur_action_sequence, file=sys.stderr)

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

            if role_change_requested:
                self.role_switches.append(unit_id)
        print(f"Step: {step} Actions: {actions}", file=sys.stderr)

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

    def _find_closest_factory(self, pos: np.array, game_state: ExtendedGameState):
        closest_distance = np.inf
        closest_factory = None

        for factory_id, factory in game_state.game_state.factories[self.player].items():
            distance = np.sum(np.abs(factory.pos - pos))
            if distance < closest_distance:
                closest_distance = distance
                closest_factory = factory_id

        return closest_factory

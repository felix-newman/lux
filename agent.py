import sys
from typing import Dict

import numpy as np

from components.actions import ActionSequence
from components.extended_game_state import ExtendedGameState
from components.extended_unit import UnitRole, ExtendedUnit
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

        self.tracked_units: Dict[str, ExtendedUnit] = dict()

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

        print(step, obs['units'][self.player], file=sys.stderr)

        # TODO factory logic and updating of coordination handler
        for factory_id, factory in game_state.game_state.factories[self.player].items():
            if factory.can_build_heavy(game_state):
                actions[factory_id] = factory.build_heavy()

        # assign tasks to units
        for unit_id, unit in game_state.game_state.units[self.player].items():
            if unit_id not in self.tracked_units:
                self.tracked_units[unit_id] = ExtendedUnit(unit=unit, unit_id=unit_id, role=UnitRole.MINER,
                                                           cur_action_sequence=ActionSequence(action_items=[], reward=0,
                                                                                              remaining_rewards=[]))

        # clean up dead units, units with empty action sequences
        for unit_id in self.tracked_units:
            if len(game_state.game_state.units[self.player][unit_id].action_queue) == 0:
                self.unit_coordination_handler.clean_up_unit(unit_id)

        for unit_id, unit in self.tracked_units.items():
            if len(game_state.game_state.units[self.player][unit_id].action_queue) == 0:
                # TODO masks are not self aware at the moment
                action_sequence = self.unit_controller.find_optimally_rewarded_action_sequence(unit,
                                                                                               self.unit_coordination_handler.occupancy_map,
                                                                                               game_state.board.rubble,
                                                                                               reward_maps=self.unit_coordination_handler.reward_action_handler,
                                                                                               real_env_step=game_state.real_env_steps)
                self.unit_coordination_handler.grant_rewards(unit_id=unit_id, action_sequence=action_sequence)
                unit.cur_action_sequence = action_sequence
                print(unit.unit.pos, unit_id, unit.cur_action_sequence, file=sys.stderr)
                actions[unit_id] = unit.cur_action_sequence.to_lux_action_queue()
        print(f"Actions: {actions}", file=sys.stderr)
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

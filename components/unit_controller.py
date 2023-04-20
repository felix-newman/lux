import math
import sys
from typing import List, Tuple

import numpy as np

from components.actions import ActionSequence, ActionItem, ActionType, Direction, RewardedAction, DIRECTION_DELTAS
from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState
from components.extended_unit import UnitRole, UnitMetadata
from components.unit_coordination_handler import UnitCoordinationHandler
from components.utils import find_top_n, get_cost_profile, transform_cost_profile, find_collision_path, get_path
from lux.unit import Unit


class UnitController:
    def __init__(self, beam_width=2):
        """Precompute the allowed reward action sequences for all unit roles"""
        self.valid_reward_sequences = dict()
        self.valid_reward_sequences[UnitRole.MINER] = [
            [ActionType.MINE_ICE, ActionType.TRANSFER_ICE, ActionType.PICKUP_POWER],
            [ActionType.PICKUP_POWER, ActionType.MINE_ICE, ActionType.TRANSFER_ICE],
        ]

        self.beam_width = beam_width
        self.day_night_cycle = self._build_day_night_cycle()

    def update_action_queue(self, unit: Unit, unit_meta: UnitMetadata, unit_coordination_handler: UnitCoordinationHandler,
                            game_state: ExtendedGameState) -> Tuple[ActionSequence, bool]:
        unit_id = unit.unit_id
        if len(unit.action_queue) == 0:
            unit_coordination_handler.clean_up_unit(unit_id)
            action_sequence = self.find_optimally_rewarded_action_sequence(unit, unit_meta,
                                                                           unit_coordination_handler.occupancy_map,
                                                                           game_state.board.rubble,
                                                                           unit_coordination_handler=unit_coordination_handler,
                                                                           real_env_step=game_state.real_env_steps)
            if action_sequence.empty:
                action_sequence = self.move_unit_to_closest_free_square(unit, unit_coordination_handler)
                return action_sequence, True
            return action_sequence, False

        next_action = unit.action_queue[0]
        if unit_coordination_handler.collision_after_lux_action(next_action, unit.pos, unit_id):
            unit_coordination_handler.clean_up_unit(unit_id)
            action_sequence = self.find_optimally_rewarded_action_sequence(unit, unit_meta,
                                                                           unit_coordination_handler.occupancy_map,
                                                                           game_state.board.rubble,
                                                                           unit_coordination_handler=unit_coordination_handler,
                                                                           real_env_step=game_state.real_env_steps)
            if action_sequence.empty:
                action_sequence = self.move_unit_to_closest_free_square(unit, unit_coordination_handler)
                return action_sequence, True
            else:
                return action_sequence, False

        return ActionSequence(action_items=[], remaining_rewards=[], reward=0), False

    @staticmethod
    def move_unit_to_closest_free_square(unit: Unit, unit_coordination_handler: UnitCoordinationHandler) -> ActionSequence:
        print(f"Move to closest free square {unit.pos}, {unit.unit_id}", file=sys.stderr)
        for direction, delta in DIRECTION_DELTAS.items():
            # new position
            new_pos = unit.pos + delta
            # check if pos on map
            if 0 < new_pos[0] < MAP_SIZE and 0 < new_pos[1] < MAP_SIZE:
                # check if pos is free
                if unit_coordination_handler.check_field_occupied(new_pos[0], new_pos[1], unit.unit_id):
                    continue
                else:
                    if direction == Direction.CENTER:
                        action_item = ActionItem(type=ActionType.MOVE_CENTER, direction=direction, amount=0, repeat=1, position=unit.pos)
                    elif direction == Direction.UP:
                        action_item = ActionItem(type=ActionType.MOVE_UP, direction=direction, amount=0, repeat=1, position=unit.pos)
                    elif direction == Direction.DOWN:
                        action_item = ActionItem(type=ActionType.MOVE_DOWN, direction=direction, amount=0, repeat=1, position=unit.pos)
                    elif direction == Direction.LEFT:
                        action_item = ActionItem(type=ActionType.MOVE_LEFT, direction=direction, amount=0, repeat=1, position=unit.pos)
                    else:
                        action_item = ActionItem(type=ActionType.MOVE_RIGHT, direction=direction, amount=0, repeat=1, position=unit.pos)

                    return ActionSequence(action_items=[action_item], remaining_rewards=[], reward=0)
        return ActionSequence(action_items=[], remaining_rewards=[], reward=-1_000_000_000)

    def find_optimally_rewarded_action_sequence(self, unit: Unit, unit_meta: UnitMetadata, occupancy_map: np.array, rubble_map: np.array,
                                                unit_coordination_handler: UnitCoordinationHandler,
                                                real_env_step: int) -> ActionSequence:

        valid_reward_sequences = self.valid_reward_sequences[unit_meta.role]

        best_sequence = ActionSequence(action_items=[], remaining_rewards=[], reward=-1_000_000_000)
        for sequence in valid_reward_sequences:
            action_sequence = self.calculate_optimal_action_sequence(unit=unit, rewarded_action_sequence=sequence,
                                                                     unit_coordination_handler=unit_coordination_handler,
                                                                     rubble_map=rubble_map,
                                                                     occupancy_map=occupancy_map, real_env_step=real_env_step)
            if action_sequence.reward > best_sequence.reward:
                best_sequence = action_sequence

        return best_sequence

    def create_candidate_sequences(self, pos: np.array, rewarded_actions: List[RewardedAction],
                                   unit_coordination_handler: UnitCoordinationHandler,
                                   occupancy_map: np.array, unit_id: str) -> List[List[Tuple[int, int]]]:
        if len(rewarded_actions) == 0:
            return [[]]

        distance_map = np.sum(np.abs(np.indices((MAP_SIZE, MAP_SIZE)) - np.array(pos)[:, None, None]), axis=0)
        discount_map = 1 - distance_map / (2 * MAP_SIZE)

        cur_action_type = rewarded_actions[0]
        discounted_reward_map = np.where(unit_coordination_handler.get_reward_mask(action_type=cur_action_type) > 0, 1, 0) * discount_map

        candidate_sequences = []
        candidates = find_top_n(self.beam_width, discounted_reward_map)
        for candidate in candidates:
            candidate_sequences += [[(candidate[0], candidate[1])] + sequence for sequence in
                                    self.create_candidate_sequences(candidate, rewarded_actions[1:], unit_coordination_handler,
                                                                    occupancy_map, unit_id)]
        return candidate_sequences

    def calculate_optimal_action_sequence(self, unit: Unit,
                                          rewarded_action_sequence: List[RewardedAction],
                                          unit_coordination_handler: UnitCoordinationHandler,
                                          rubble_map: np.array, occupancy_map: np.array,
                                          real_env_step: int) -> ActionSequence:
        unit_charge = 10 if unit.unit_type == 'HEAVY' else 1  # TODO import constant here
        move_costs = 20 if unit.unit_type == 'HEAVY' else 1  # same here
        digging_costs = 60 if unit.unit_type == 'HEAVY' else 5  # and here
        digging_speed = 20 if unit.unit_type == 'HEAVY' else 2  # and here
        battery_capacity = 3000 if unit.unit_type == 'HEAVY' else 150  # and here

        # TODO issue with factory actions: Since factories are spread out there are multiple tiles which actually dont
        # differ reward wise. I would be better to treat them as one field, so that multiple factories would be
        # considered in the search
        position_sequences = self.create_candidate_sequences(unit.pos, rewarded_action_sequence,
                                                             unit_coordination_handler, occupancy_map, unit.unit_id)
        best_action_sequence = ActionSequence(action_items=[], reward=-1_000_000_000, remaining_rewards=[])
        for sequence in position_sequences:
            sequence = [(unit.pos[0], unit.pos[1])] + sequence
            segments = list(zip(sequence, sequence[1:]))
            segment_waypoints = [find_collision_path(mask=occupancy_map, start=segments[0][0], end=segments[0][1])] + [
                get_path(segment[0], segment[1]) for segment in segments[1:]]
            if segment_waypoints[0] is None or (
                    segment_waypoints[0] == [] and occupancy_map[unit.pos[0], unit.pos[1]] != unit.unit_id.split('_')[1]):
                continue

            segment_cost_profiles = [get_cost_profile(positions=np.array(waypoints), cost_map=rubble_map) for waypoints in
                                     segment_waypoints]
            segment_cost_profiles = [transform_cost_profile(cost_profile, unit_type=unit.unit_type) for cost_profile in
                                     segment_cost_profiles]
            power_profiles = []
            power_start = unit.power - 10  # TODO cost for updating the action queue
            for (cost_profile, following_rewarded_action) in zip(segment_cost_profiles, rewarded_action_sequence):
                power_profile = power_start + np.cumsum(
                    -cost_profile + unit_charge * self.day_night_cycle[real_env_step:real_env_step + cost_profile.shape[0]])
                if np.any(power_profile < 0):
                    break
                power_profiles.append(power_profile)

                power_end = power_profile[-1]
                # TODO consider edge cases with self destruction
                if following_rewarded_action == ActionType.PICKUP_POWER:
                    # TODO this should be taking into account the power situation of the factory
                    power_end = power_end + 50 if unit.unit_type == 'LIGHT' else power_end + 200
                power_start = power_end
            else:
                power_for_digging = self._power_for_digging(power_profiles=power_profiles,
                                                            rewarded_action_sequence=rewarded_action_sequence)
                safety_moves = 2  # TODO collect free parameters in central place
                power_for_digging -= safety_moves * move_costs
                if power_for_digging < digging_costs and (
                        ActionType.MINE_ORE in rewarded_action_sequence or ActionType.MINE_ICE in rewarded_action_sequence):
                    continue

                # build sequence
                reward = 0
                action_items: List[ActionItem] = []
                cur_ice = unit.cargo.ice
                cur_ore = unit.cargo.ore
                cur_pos = unit.pos
                for (waypoints, power_profile, following_rewarded_action) in zip(segment_waypoints, power_profiles,
                                                                                 rewarded_action_sequence):
                    action_items += self._translate_waypoints_to_actions(start=cur_pos, waypoints=waypoints,
                                                                         occupancy_map=occupancy_map)  # TODO care for collisions
                    # TODO compute repeat variable and set it in the action item afterwards would be better. Should be found by optimization
                    # over the reward function
                    cur_pos = waypoints[-1] if len(waypoints) > 0 else cur_pos
                    repeat = 1
                    amount = 0
                    if following_rewarded_action is ActionType.MINE_ORE:
                        repeat = math.floor(power_for_digging / digging_costs)  # repeat refers to the number of additional actions
                        cur_ore += repeat * digging_speed
                    if following_rewarded_action is ActionType.MINE_ICE:
                        repeat = math.floor(power_for_digging / digging_costs)  # repeat refers to the number of additional actions
                        cur_ice += repeat * digging_speed

                    if following_rewarded_action is ActionType.PICKUP_POWER:
                        amount = 3000 if unit.unit_type == 'HEAVY' else 70  # TODO optimize amount of picked up power

                    rewarded_action = ActionItem(type=following_rewarded_action, position=np.array(cur_pos),
                                                 repeat=repeat, direction=Direction.CENTER, amount=amount)

                    new_reward = self.calculate_reward(action_item=rewarded_action, cur_ice=cur_ice, cur_ore=cur_ore,
                                                       cur_power=power_profile[-1],
                                                       unit_coordination_handler=unit_coordination_handler,
                                                       battery_capacity=battery_capacity)
                    rewarded_action.reward = new_reward
                    reward += new_reward
                    action_items.append(rewarded_action)

                # TODO compute remaining rewards, should be done in calculate_reward()
                reward -= len(action_items) * 5  # TODO this is a hack to make the robot not do too many actions
                action_sequence = ActionSequence(action_items=action_items, reward=reward, remaining_rewards=[0])

                if action_sequence.estimate_lux_action_queue_length() < 20:
                    if action_sequence.reward > best_action_sequence.reward:
                        best_action_sequence = action_sequence

        return best_action_sequence

    @staticmethod
    def _build_day_night_cycle():
        return np.array(([1] * 30 + [0] * 20) * 30)  # TODO currently extends into the future

    @staticmethod
    def calculate_reward(action_item: ActionItem, unit_coordination_handler: UnitCoordinationHandler, cur_ice: int, cur_ore: int,
                         cur_power: int, battery_capacity: int) -> float:
        action_type = action_item.type
        x, y = action_item.position

        if action_type is ActionType.MINE_ICE:
            return unit_coordination_handler.reward_action_handler[action_type].actual_reward_map[x, y] * action_item.repeat
        if action_type is ActionType.MINE_ORE:
            return unit_coordination_handler.reward_action_handler[action_type].actual_reward_map[x, y] * action_item.repeat
        if action_type is ActionType.TRANSFER_ORE:
            return unit_coordination_handler.reward_action_handler[action_type].actual_reward_map[x, y] * cur_ore
        if action_type is ActionType.TRANSFER_ICE:
            return unit_coordination_handler.reward_action_handler[action_type].actual_reward_map[x, y] * cur_ice
        if action_type is ActionType.PICKUP_POWER:
            return 0.001 * (battery_capacity - cur_power)
        else:
            print("Warning: invalid action type for reward", file=sys.stderr)
        return 0

    @staticmethod
    def _translate_waypoints_to_actions(start: np.array, waypoints: List[Tuple[int, int]], occupancy_map: np.array) -> List[ActionItem]:
        action_items = []
        cur_pos = start
        for waypoint in waypoints:
            dx = waypoint[0] - cur_pos[0]
            dy = waypoint[1] - cur_pos[1]

            if dx == 1:
                action_items.append(
                    ActionItem(type=ActionType.MOVE_RIGHT, position=cur_pos, repeat=1, direction=Direction.CENTER, amount=0))
            elif dx == -1:
                action_items.append(ActionItem(type=ActionType.MOVE_LEFT, position=cur_pos, repeat=1, direction=Direction.CENTER, amount=0))
            elif dy == 1:
                action_items.append(ActionItem(type=ActionType.MOVE_DOWN, position=cur_pos, repeat=1, direction=Direction.CENTER, amount=0))
            elif dy == -1:
                action_items.append(ActionItem(type=ActionType.MOVE_UP, position=cur_pos, repeat=1, direction=Direction.CENTER, amount=0))
            else:
                print("ERROR: Invalid waypoint sequence")

            cur_pos = waypoint

        return action_items

    @staticmethod
    def _power_for_digging(power_profiles: List[np.array], rewarded_action_sequence: List[RewardedAction]):
        try:
            digging_idx = rewarded_action_sequence.index(ActionType.MINE_ICE)
        except ValueError:
            try:
                digging_idx = rewarded_action_sequence.index(ActionType.MINE_ORE)
            except ValueError:
                return 0

        return np.min(np.concatenate(power_profiles[digging_idx + 1:]))

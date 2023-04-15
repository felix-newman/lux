import math
from typing import Dict, List, Tuple

import numpy as np

from components.actions import ActionSequence, RewardedActionSequence, RewardedActionType, ActionItem, ActionType, Direction
from components.constants import MAP_SIZE
from components.extended_unit import ExtendedUnit, UnitRole
from components.unit_coordination_handler import RewardActionHandler
from components.utils import create_sequences, find_top_n, get_path, get_cost_profile


class UnitController:
    def __init__(self, beam_width=2):
        """Precompute the allowed reward action sequences for all unit roles"""
        self.valid_reward_sequences = dict()
        self.valid_reward_sequences[UnitRole.MINER] = [RewardedActionSequence(
            [RewardedActionType.MINE_ICE, RewardedActionType.TRANSFER_ICE, RewardedActionType.PICKUP_POWER])]

        self.beam_width = beam_width
        self.day_night_cycle = self._build_day_night_cycle()

    def find_optimally_rewarded_action_sequence(self, unit: ExtendedUnit, occupancy_map: np.array, rubble_map: np.array,
                                                reward_maps: Dict[RewardedActionType, RewardActionHandler],
                                                real_env_step: int) -> ActionSequence:

        valid_reward_sequences = self.valid_reward_sequences[unit.role]
        best_sequence = None
        best_reward = -1
        for sequence in valid_reward_sequences:
            action_sequence = self.calculate_optimal_action_sequence(unit=unit, rewarded_action_sequence=sequence,
                                                                     reward_maps=reward_maps, rubble_map=rubble_map,
                                                                     occupancy_map=occupancy_map, real_env_step=real_env_step)
            if action_sequence.reward > best_reward:
                best_sequence = action_sequence
                best_reward = action_sequence.reward

        return best_sequence

    def calculate_optimal_action_sequence(self, unit: ExtendedUnit,
                                          rewarded_action_sequence: RewardedActionSequence, reward_maps: Dict[
                RewardedActionType, RewardActionHandler], rubble_map: np.array, occupancy_map: np.array,
                                          real_env_step: int) -> ActionSequence:
        # TODO issue with factory actions: Since factories are spread out there are multiple tiles which actually dont
        # differ reward wise. I would be better to treat them as one field, so that multiple factories would be
        # considered in the search

        distance_map = np.sum(np.abs(np.indices((MAP_SIZE, MAP_SIZE)) - np.array(unit.unit.pos)[:, None, None]), axis=0)
        discount_map = -2 * distance_map / MAP_SIZE

        discounted_reward_maps = [np.where(reward_maps[action].reward_mask != 0, 1, 0) * discount_map for action in
                                  rewarded_action_sequence.action_items]
        # TODO candidate generation should take into consideration the current position of the unit in the sequence,
        # right now it assumes a constant starting point
        candidates = [find_top_n(2, discounted_map) for discounted_map in discounted_reward_maps]

        position_sequences = create_sequences(candidates)
        best_action_sequence = None

        for sequence in position_sequences:
            # TODO careful what happens when already on the field where the action is supposed to take place
            # create action sequence, estimate power consumption
            sequence = [(unit.unit.pos[0], unit.unit.pos[1])] + sequence
            segments = list(zip(sequence, sequence[1:]))
            segment_waypoints = [get_path(segment[0], segment[1]) for segment in segments]
            segment_cost_profiles = [get_cost_profile(positions=waypoints, cost_map=rubble_map) for waypoints in segment_waypoints]
            power_profiles = []

            power_start = unit.unit.power
            power_end = None
            unit_charge = 10 if unit.unit.unit_type == 'HEAVY' else 1  # TODO import constant here
            move_costs = 20 if unit.unit.unit_type == 'HEAVY' else 1  # same here
            digging_costs = 60 if unit.unit.unit_type == 'HEAVY' else 5  # and here
            digging_speed = 20 if unit.unit.unit_type == 'HEAVY' else 2  # and here
            battery_capacity = 3000 if unit.unit.unit_type == 'HEAVY' else 100  # and here

            safety_moves = 2  # TODO collect free parameters in central place
            for cost_profile in segment_cost_profiles:
                # TODO simplification. For light units there is actually a floor() operation involved, which leads this to
                # be an upper bound
                power_profile = power_start + np.cumsum(
                    cost_profile + unit_charge * self.day_night_cycle[real_env_step:real_env_step + cost_profile.shape[0]])
                if np.any(power_profile < 0):
                    # not valid sequence, robot runs out of power
                    break
                power_profiles.append(power_profile)
                power_end = power_profile[-1]
                power_start = power_end
            else:
                # TODO consider edge cases with self destruction
                # robot does not run out of power during moving
                power_for_digging = power_end - safety_moves * move_costs
                if power_for_digging < digging_costs:
                    continue

            # build sequence
            reward = 0
            action_items: List[ActionItem] = []
            cur_ice = unit.unit.cargo.ice
            cur_ore = unit.unit.cargo.ore
            for (waypoints, power_profile, following_rewarded_action) in zip(segment_waypoints, power_profiles,
                                                                             rewarded_action_sequence.action_items):
                action_items += self._translate_waypoints_to_actions(start=unit.unit.pos, waypoints=waypoints,
                                                                     occupancy_map=occupancy_map)  # TODO care for collisions
                # TODO compute repeat variable and set it in the action item afterwards would be better. Should be found by optimization
                # over the reward function
                repeat = 1
                if following_rewarded_action is RewardedActionType.MINE_ORE:
                    repeat = math.floor(power_for_digging / digging_costs)
                    cur_ore += repeat * digging_speed
                if following_rewarded_action is RewardedActionType.MINE_ICE:
                    repeat = math.floor(power_for_digging / digging_costs)
                    cur_ice += repeat * digging_speed

                rewarded_action = ActionItem(type=following_rewarded_action.to_action_type(), position=action_items[-1].position,
                                             repeat=repeat, direction=Direction.CENTER)
                action_items += rewarded_action

                reward += self.calculate_reward(action_item=rewarded_action, cur_ice=cur_ice, cur_ore=cur_ore, cur_power=power_profile[-1],
                                                reward_maps=reward_maps, battery_capacity=battery_capacity)

            action_sequence = ActionSequence(action_items=action_items, reward=reward,
                                             remaining_rewards=[0])  # TODO compute remaining rewards, should be done in calculate_reward()
            if best_action_sequence is None or action_sequence.reward > best_action_sequence.reward:
                best_action_sequence = action_sequence

        return best_action_sequence

    @staticmethod
    def _build_day_night_cycle():
        return np.array(([1] * 30 + [0] * 20) * 20)

    @staticmethod
    def calculate_reward(action_item: ActionItem, reward_maps: Dict[RewardedActionType, RewardActionHandler], cur_ice: int, cur_ore: int,
                         cur_power: int, battery_capacity: int) -> int:
        action_type = action_item.type.to_rewarded_action_type()
        reward_map = reward_maps[action_type]
        reward = reward_map.reward_map[action_item.position[0], action_item.position[1]]
        if action_type is RewardedActionType.MINE_ICE:
            return reward * action_item.repeat
        if action_type is RewardedActionType.MINE_ORE:
            return reward * action_item.repeat
        if action_type is RewardedActionType.TRANSFER_ORE:
            return reward * cur_ore
        if action_type is RewardedActionType.TRANSFER_ICE:
            return reward * cur_ice
        if action_type is RewardedActionType.PICKUP_POWER:
            return reward * (battery_capacity - cur_power)
        return reward

    @staticmethod
    def _translate_waypoints_to_actions(start: np.array, waypoints: List[Tuple[int, int]], occupancy_map: np.array) -> List[
        ActionItem]:
        action_items = []
        cur_pos = start
        for waypoint in waypoints:
            dx = waypoint[0] - cur_pos[0]
            dy = waypoint[1] - cur_pos[1]

            if dx == 1:
                action_items.append(ActionItem(type=ActionType.MOVE_RIGHT, position=cur_pos, repeat=1, direction=Direction.CENTER))
            elif dx == -1:
                action_items.append(ActionItem(type=ActionType.MOVE_LEFT, position=cur_pos, repeat=1, direction=Direction.CENTER))
            elif dy == 1:
                action_items.append(ActionItem(type=ActionType.MOVE_DOWN, position=cur_pos, repeat=1, direction=Direction.CENTER))
            elif dy == -1:
                action_items.append(ActionItem(type=ActionType.MOVE_UP, position=cur_pos, repeat=1, direction=Direction.CENTER))

        return action_items

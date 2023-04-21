import numpy as np
import copy

from components.actions import ActionItem, ActionSequence, ActionType, Direction, rewarded_actions_from_lux_action_queue


class TestActionItem:
    def test_repeat_option(self):
        action_sequence = ActionSequence(
            action_items=[ActionItem(type=ActionType.MOVE_UP, position=(0, 0), repeat=1, amount=0, direction=Direction.UP),
                          ActionItem(type=ActionType.MOVE_UP, position=(0, 0), repeat=1, amount=0, direction=Direction.UP)
                          ], reward=0,
            remaining_rewards=[0])

        original_seq = copy.deepcopy(action_sequence)
        result = action_sequence.to_lux_action_queue()
        expected_result = [np.array([0, 1, 0, 0, 0, 2])]
        assert np.array_equal(result, expected_result)
        assert action_sequence == original_seq

    def test_two_times_same_action_different_positions(self):
        action_sequence = ActionSequence(
            action_items=[ActionItem(type=ActionType.DIG, position=(0, 1), repeat=1, amount=0, direction=Direction.UP),
                          ActionItem(type=ActionType.DIG, position=(0, 2), repeat=1, amount=0, direction=Direction.UP)
                          ], reward=0,
            remaining_rewards=[0])

        original_seq = copy.deepcopy(action_sequence)
        result = action_sequence.to_lux_action_queue()
        expected_result = [np.array([3, 0, 0, 0, 0, 1]), np.array([3, 0, 0, 0, 0, 1])]
        assert np.array_equal(result, expected_result)
        assert action_sequence == original_seq

    def test_exclude_return_actions(self):
        action_sequence = ActionSequence(
            action_items=[ActionItem(type=ActionType.MOVE_UP, position=(0, 0), repeat=1, amount=0, direction=Direction.UP),
                          ActionItem(type=ActionType.RETURN, position=(0, 0), repeat=1, amount=0, direction=Direction.UP)
                          ], reward=0,
            remaining_rewards=[0])

        original_seq = copy.deepcopy(action_sequence)
        result = action_sequence.to_lux_action_queue()
        expected_result = [np.array([0, 1, 0, 0, 0, 1])]
        assert np.array_equal(result, expected_result)
        assert action_sequence == original_seq


class TestRewardedActionsFromLuxQueue:
    def test_recognize_dig(self):
        lux_queue = [np.array([2, 0, 4, 35, 0, 1]), np.array([0, 3, 0, 0, 0, 3]), np.array([3, 0, 0, 0, 0, 4])]
        result = rewarded_actions_from_lux_action_queue(lux_queue)
        expected_result = [ActionType.PICKUP_POWER, ActionType.DIG]
        assert result == expected_result

    def test_recognize_ice(self):
        lux_queue = [np.array([0, 1, 0, 0, 0, 2]), np.array([3, 0, 0, 0, 0, 6]), np.array([0, 3, 0, 0, 0, 1]),
                     np.array([1, 0, 0, 3000, 0, 1])]
        result = rewarded_actions_from_lux_action_queue(lux_queue)
        expected_result = [ActionType.MINE_ICE, ActionType.TRANSFER_ICE]
        assert result == expected_result

    def test_recognize_ore(self):
        lux_queue = [np.array([0, 1, 0, 0, 0, 2]), np.array([3, 0, 0, 0, 0, 6]), np.array([0, 3, 0, 0, 0, 1]),
                     np.array([1, 0, 1, 3000, 0, 1])]
        result = rewarded_actions_from_lux_action_queue(lux_queue)
        expected_result = [ActionType.MINE_ORE, ActionType.TRANSFER_ORE]
        assert result == expected_result

    def test_insert_return_action_when_last_action_is_move(self):
        lux_queue = [np.array([0, 1, 0, 0, 0, 2]), np.array([3, 0, 0, 0, 0, 6]), np.array([0, 1, 0, 0, 0, 2])]
        result = rewarded_actions_from_lux_action_queue(lux_queue)
        expected_result = [ActionType.DIG, ActionType.RETURN]
        assert result == expected_result

    def test_does_not_change_original_queue(self):
        lux_queue = [np.array([0, 1, 0, 0, 0, 2]), np.array([3, 0, 0, 0, 0, 6]), np.array([0, 3, 0, 0, 0, 1]),
                     np.array([1, 0, 0, 3000, 0, 1])]

        original_queue = copy.deepcopy(lux_queue)
        _ = rewarded_actions_from_lux_action_queue(lux_queue)
        assert np.array_equal(lux_queue, original_queue)

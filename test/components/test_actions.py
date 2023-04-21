import numpy as np
import copy

from components.actions import ActionItem, ActionSequence, ActionType, Direction


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



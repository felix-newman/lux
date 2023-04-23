from components.actions import ActionType, rewarded_actions_from_lux_action_queue
from components.extended_unit import UnitMetadata, UnitRole
from lux.unit import Unit


class RewardSequenceCalculator:
    def __init__(self):
        pass

    def calculate_valid_reward_sequence(self, unit: Unit, unit_meta: UnitMetadata):
        if unit_meta.role == UnitRole.MINER:
            return self.calculate_miner_reward_sequences(unit)
        elif unit_meta.role == UnitRole.DIGGER:
            return self.calculate_digger_reward_sequences(unit)

    @staticmethod
    def calculate_miner_reward_sequences(unit: Unit):
        rewarded_actions = rewarded_actions_from_lux_action_queue(unit.action_queue)
        if len(rewarded_actions) > 0 and rewarded_actions != [ActionType.RETURN]:
            return [rewarded_actions]

        valid_reward_sequences = [
            [ActionType.PICKUP_POWER, ActionType.MINE_ICE, ActionType.TRANSFER_ICE],
            [ActionType.PICKUP_POWER, ActionType.MINE_ORE, ActionType.TRANSFER_ORE],
            [ActionType.RECHARGE, ActionType.RETURN]
        ]

        if unit.cargo.ice > 0:
            valid_reward_sequences.append([ActionType.TRANSFER_ICE])
            valid_reward_sequences.append([ActionType.MINE_ICE, ActionType.TRANSFER_ICE])
        elif unit.cargo.ore > 0:
            valid_reward_sequences.append([ActionType.TRANSFER_ORE])
            valid_reward_sequences.append([ActionType.MINE_ORE, ActionType.TRANSFER_ORE])

        return valid_reward_sequences

    @staticmethod
    def calculate_digger_reward_sequences(unit: Unit):
        rewarded_actions = rewarded_actions_from_lux_action_queue(unit.action_queue)
        if len(rewarded_actions) > 0 and rewarded_actions != [ActionType.RETURN]:
            return [rewarded_actions]

        valid_reward_sequences = [
            [ActionType.PICKUP_POWER, ActionType.DIG, ActionType.RETURN],
            [ActionType.PICKUP_POWER, ActionType.DIG, ActionType.DIG, ActionType.RETURN],
            [ActionType.DIG, ActionType.DIG, ActionType.RETURN],
            [ActionType.RECHARGE, ActionType.RETURN]
        ]

        return valid_reward_sequences



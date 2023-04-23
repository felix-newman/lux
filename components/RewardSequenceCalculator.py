from components.actions import ActionType, rewarded_actions_from_lux_action_queue
from components.extended_unit import UnitMetadata, UnitRole
from components.unit_coordination_handler import UnitCoordinationHandler
from lux.unit import Unit


class RewardSequenceCalculator:
    def __init__(self):
        pass

    def calculate_valid_reward_sequence(self, unit: Unit, unit_meta: UnitMetadata, unit_coordination_handler: UnitCoordinationHandler):
        if unit_meta.role == UnitRole.MINER:
            return self.calculate_miner_reward_sequences(unit, unit_coordination_handler)
        elif unit_meta.role == UnitRole.DIGGER:
            return self.calculate_digger_reward_sequences(unit, unit_coordination_handler)

    def calculate_miner_reward_sequences(self, unit: Unit, unit_coordination_handler: UnitCoordinationHandler):
        if unit_coordination_handler.on_fight_field(unit.pos):
            return self.worker_behavior_on_enemy_encounter(unit, unit_coordination_handler)

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

    def calculate_digger_reward_sequences(self, unit: Unit, unit_coordination_handler: UnitCoordinationHandler):
        if unit_coordination_handler.on_fight_field(unit.pos):
            return self.worker_behavior_on_enemy_encounter(unit, unit_coordination_handler)

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

    @staticmethod
    def worker_behavior_on_enemy_encounter(unit: Unit, unit_coordination_handler: UnitCoordinationHandler):
        heaviest_robot, max_power_value = unit_coordination_handler.get_strongest_enemy(unit.pos)
        own_type = 2 if unit.unit_type == 'HEAVY' else 1
        if heaviest_robot == 1 and unit.unit_type == 'HEAVY':
            return None
        else:
            adjusted_power = unit.power - 5 if unit.unit_type == 'LIGHT' else unit.power - 80
            if own_type < heaviest_robot or (own_type == heaviest_robot and adjusted_power < max_power_value):
                return [[ActionType.RETURN]]  # TODO should be flee
            else:
                return [[ActionType.FIGHT, ActionType.RETURN], [ActionType.RETURN]]

    # TODO for fighter: consider both cases of staying and moving to fight field


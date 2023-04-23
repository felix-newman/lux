from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from components.constants import MAP_SIZE
from components.extended_game_state import ExtendedGameState
from lux.unit import Unit


class EnemyMap:
    def __init__(self, opp_player: str):
        self.enemy_map: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.power_map: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        self.opp_player = opp_player
        self.fight_map = np.zeros((MAP_SIZE, MAP_SIZE))

    def update(self, game_state: ExtendedGameState):
        self.enemy_map = np.zeros((MAP_SIZE, MAP_SIZE))
        self.power_map: np.array = np.zeros((MAP_SIZE, MAP_SIZE))
        for _, unit in game_state.game_state.units[self.opp_player].items():
            self.enemy_map[unit.pos[0], unit.pos[1]] = 1 if unit.unit_type == 'LIGHT' else 2
            self.power_map[unit.pos[0], unit.pos[1]] = unit.power

        self.fight_map = binary_dilation(self.enemy_map) * 1000

    def is_fighting_position(self, x, y):
        return self.fight_map[x, y] > 0

    def superior_enemies_map(self, unit: Unit) -> np.array:
        if unit.unit_type == 'LIGHT':
            heavy_units = self.enemy_map == 2
            lights_with_more_power = np.logical_and(self.enemy_map == 1, self.power_map > unit.power - 5) # 5 = 1 queue cost + 1 move cost + 3 avg rubble cost
            superior_enemies = np.logical_or(heavy_units, lights_with_more_power)
        else:
            superior_enemies = np.logical_and(self.enemy_map == 2, self.power_map > unit.power - 80) # 80 = 10 queue cost + 20 move cost + 50 avg rubble cost

        return binary_dilation(superior_enemies)

    # TODO atm also considers diagonal enemies
    def get_strongest_enemy(self, pos: np.array) -> Tuple[int, int]:
        window = self.enemy_map[max(0, pos[0] - 1):min(self.enemy_map.shape[0], pos[0] + 2), max(0, pos[1] - 1):min(self.enemy_map.shape[1], pos[1] + 2)]

        # Find the positions of the maximum value(s) in the window
        heaviest_robot = np.max(window)
        positions_of_heaviest_robots = np.argwhere(window == heaviest_robot)

        # Get the maximum value(s) in arr2 corresponding to the max position(s)
        power_values = [self.power_map[max(0, pos[0] - 1) + p[0], max(0, pos[1] - 1) + p[1]] for p in positions_of_heaviest_robots]

        # Find the overall maximum value in arr2
        max_power_value = np.max(power_values)

        return heaviest_robot, max_power_value

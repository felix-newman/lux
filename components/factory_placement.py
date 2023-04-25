import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage import binary_dilation

from components.utils import normalize_matrix
from lux.kit import GameState


def compute_lichen_potential_field(x, y, rubble_map, dig_speed=10.0, steps=10):
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.int32)

    cur_rubble = np.copy(rubble_map)
    cur_rubble[x, y] = 0

    lichen = np.zeros_like(rubble_map)
    lichen[x, y] = 1

    for step in range(steps):
        work_area = binary_dilation(lichen, structure)
        cur_rubble = np.clip(cur_rubble - work_area * dig_speed, 0, 100)
        lichen = np.logical_and(work_area, cur_rubble == 0)

    return np.sum(lichen)


def compute_lichen_potential_map(rubble_map, dig_speed=5, steps=20):
    lichen_potential = np.zeros_like(rubble_map)
    for x in range(lichen_potential.shape[1]):
        for y in range(lichen_potential.shape[0]):
            lichen_potential[x, y] = compute_lichen_potential_field(x, y, rubble_map, dig_speed=dig_speed, steps=steps)

    return lichen_potential


def compute_factory_value_map(game_state: GameState, lichen_potential_map: np.array, dig_speed=5, steps=5):
    rubble_map = game_state.board.rubble
    ice_map = game_state.board.ice
    ore_map = game_state.board.ore

    occupancy_map = np.where(game_state.board.factory_occupancy_map >= 0, 1, 0)
    dilated_occupancy = binary_dilation(occupancy_map, np.ones((3, 3)))

    valid_ice_map = ice_map * (1 - dilated_occupancy)
    valid_ore_map = ore_map * (1 - dilated_occupancy)


    ice_distance = distance_transform_cdt(1 - valid_ice_map, metric='taxicab')
    ore_distance = distance_transform_cdt(1 - valid_ore_map, metric='taxicab')

    if np.max(ice_distance) == -1 and np.min(ice_distance) == -1:
        ice_distance = distance_transform_cdt(1 - ice_map, metric='taxicab')
    ice_score_map = normalize_matrix(96 - ice_distance)

    if np.max(ore_distance) == -1 and np.min(ore_distance) == -1:
        ore_distance = distance_transform_cdt(1 - ore_map, metric='taxicab')
    ore_score_map = normalize_matrix(96 - ore_distance)

    return 8 * ice_score_map + 2 * lichen_potential_map + ore_score_map

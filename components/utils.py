from typing import List, Tuple

import numpy as np
import networkx as nx


def normalize_matrix(matrix):
    """
    This function normalizes a numpy matrix to a range of 0 to 1.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def find_top_n(n: int, arr: np.array) -> List[Tuple[int, int]]:
    """
    Finds the top n elements in a numpy array and returns their indices as a list of tuples.
    """
    indices = np.argpartition(arr, -n, axis=None)[-n:]

    # convert the flattened indices to 2D indices
    indices_2d = np.unravel_index(indices, arr.shape)

    # convert the row and column indices to a list of 2D coordinates (as tuples)
    return list(zip(indices_2d[0], indices_2d[1]))


def create_sequences(candidates: List[List[Tuple[int, int]]]):
    """
    Creates a list of all possible sequences when taking the first element from the first nested list, the second
    from the second etc.

    """
    if not candidates:
        return [[]]
    else:
        result = []
        for item in candidates[0]:
            for seq in create_sequences(candidates[1:]):
                result.append([item] + seq)
        return result


def get_path(start, end) -> List[Tuple[int, int]]:
    """Returns a list of points that connects the start and endpoint. Does not contain the start point."""
    path = []
    x, y = start
    dx, dy = end[0] - x, end[1] - y
    abs_dx, abs_dy = abs(dx), abs(dy)
    x_step = 1 if dx > 0 else -1
    y_step = 1 if dy > 0 else -1
    max_delta = max(abs_dx, abs_dy)
    for i in range(abs_dx):
        x += x_step
        path.append((x, y))
    for i in range(abs_dy):
        y += y_step
        path.append((x, y))
    return path


def find_collision_path(mask: np.array, start, end) -> List[Tuple[int, int]]:
    if mask[start] == 1 or mask[end] == 1:
        return None

    path = []
    visited = {start}
    current = start
    while current != end:
        # Find neighbors of current position
        neighbors = []
        if current[0] > 0 and mask[current[0] - 1, current[1]] == 0 and (current[0] - 1, current[1]) not in visited:
            neighbors.append((current[0] - 1, current[1]))
        if current[0] < mask.shape[0] - 1 and mask[current[0] + 1, current[1]] == 0 and (current[0] + 1, current[1]) not in visited:
            neighbors.append((current[0] + 1, current[1]))
        if current[1] > 0 and mask[current[0], current[1] - 1] == 0 and (current[0], current[1] - 1) not in visited:
            neighbors.append((current[0], current[1] - 1))
        if current[1] < mask.shape[1] - 1 and mask[current[0], current[1] + 1] == 0 and (current[0], current[1] + 1) not in visited:
            neighbors.append((current[0], current[1] + 1))

        # Choose the neighbor closest to the endpoint
        if neighbors:
            neighbor_distances = [np.linalg.norm(np.array(neighbor) - np.array(end)) for neighbor in neighbors]
            closest_neighbor_index = np.argmin(neighbor_distances)
            closest_neighbor = neighbors[closest_neighbor_index]
            path.append(closest_neighbor)
            visited.add(closest_neighbor)
            current = closest_neighbor
        else:
            # If no valid neighbors, backtrack to the last unexplored position
            path.pop()
            if not path:
                return None
            current = path[-1]

    return path


def get_cost_profile(positions: np.array, cost_map: np.array) -> np.array:
    """Returns the total cost of the positions given in the first list."""
    if not np.any(positions):
        return np.array([0])
    row_indices, col_indices = positions[:, 0], positions[:, 1]
    return cost_map[row_indices, col_indices]


def transform_cost_profile(cost_profile: np.array, unit_type: str) -> np.array:
    """Transforms the cost profile according to the unit type."""
    if unit_type == 'HEAVY':
        return np.floor(cost_profile + 20)
    elif unit_type == 'LIGHT':
        return np.floor(1 + cost_profile * 0.05)


def build_travel_graph(cost_map: np.array):
    g = nx.DiGraph()
    add_delta = lambda a: tuple(np.array(a[0]) + np.array(a[1]))

    for x in range(cost_map.shape[0]):
        for y in range(cost_map.shape[1]):
            g.add_node((x, y), rubble=cost_map[x, y])

    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for g1 in g.nodes:
        for delta in deltas:
            g2 = add_delta((g1, delta))
            if g.has_node(g2):
                g.add_edge(g1, g2, cost=20 + cost_map[g2])

    return g

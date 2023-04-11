import numpy as np


def normalize_matrix(matrix):
    """
    This function normalizes a numpy matrix to a range of 0 to 1.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix
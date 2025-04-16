import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def clean_matrix(matrix: np.ndarray, precision: float = 1e-5) -> np.ndarray:
    """Clean numerical matrix using symmetry precision"""
    cleaned = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cleaned[i, j] = round_to_symmetry(matrix[i, j], precision)
    return cleaned


def round_to_symmetry(value: float, precision: float) -> float:
    """Round value to nearest symmetric fraction"""
    d = Decimal(str(value)).quantize(Decimal(str(precision)), rounding=ROUND_HALF_UP)
    return float(d)

"""Ordinary least squares via normal equations with partial pivoting."""

from .errors import TidalError

_RANK_TOL = 1e-10


def solve_least_squares(design, target):
    """Solve min ||A x - b|| for a full-column-rank matrix A.

    Returns the coefficient vector. Raises TidalError if the design
    matrix is rank deficient (no unique fit is possible).
    """
    rows = len(design)
    cols = len(design[0])
    if rows < cols:
        raise TidalError(
            f"cannot fit {cols} parameters from {rows} observations "
            f"(need at least {cols})"
        )
    normal = [[0.0] * cols for _ in range(cols)]
    rhs = [0.0] * cols
    for row, value in zip(design, target):
        for i in range(cols):
            rhs[i] += row[i] * value
            ri = row[i]
            for j in range(i, cols):
                normal[i][j] += ri * row[j]
    for i in range(cols):
        for j in range(i):
            normal[i][j] = normal[j][i]
    return _solve_symmetric(normal, rhs)


def _solve_symmetric(matrix, rhs):
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    scale = [max(abs(v) for v in matrix[i]) or 1.0 for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]) / scale[r])
        if abs(aug[pivot][col]) <= _RANK_TOL * scale[pivot]:
            raise TidalError(
                "design matrix is rank deficient; observations do not "
                "uniquely determine all constituents"
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = 1.0 / aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = aug[i][n] - sum(aug[i][j] * result[j] for j in range(i + 1, n))
        result[i] = acc / aug[i][i]
    return result

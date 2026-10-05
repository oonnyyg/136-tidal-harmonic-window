"""Small dense linear algebra helpers (standard library only)."""

from __future__ import annotations


def solve_least_squares(design: list[list[float]], values: list[float],
                        tol: float = 1e-10) -> list[float]:
    """Solve min ||A x - b|| via normal equations with partial pivoting.

    Raises ValueError if the design matrix is rank deficient, i.e. the
    coefficients cannot be determined uniquely.
    """
    n_rows = len(design)
    if n_rows == 0:
        raise ValueError("no observations provided")
    n_cols = len(design[0])
    if any(len(row) != n_cols for row in design):
        raise ValueError("ragged design matrix")
    if n_rows < n_cols:
        raise ValueError(
            f"underdetermined system: {n_rows} observations for "
            f"{n_cols} unknowns")

    # Normal equations: (A^T A) x = A^T b
    ata = [[0.0] * n_cols for _ in range(n_cols)]
    atb = [0.0] * n_cols
    for row, value in zip(design, values):
        for i in range(n_cols):
            ri = row[i]
            atb[i] += ri * value
            for j in range(i, n_cols):
                ata[i][j] += ri * row[j]
    for i in range(n_cols):
        for j in range(i):
            ata[i][j] = ata[j][i]

    # Gaussian elimination with partial pivoting on A^T A.
    scale = max(1.0, max(max(abs(v) for v in row) for row in ata))
    for col in range(n_cols):
        pivot = max(range(col, n_cols), key=lambda r: abs(ata[r][col]))
        if abs(ata[pivot][col]) <= tol * scale:
            raise ValueError(
                "design matrix is rank deficient; cannot fit uniquely")
        if pivot != col:
            ata[col], ata[pivot] = ata[pivot], ata[col]
            atb[col], atb[pivot] = atb[pivot], atb[col]
        inv = 1.0 / ata[col][col]
        for r in range(col + 1, n_cols):
            factor = ata[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n_cols):
                ata[r][c] -= factor * ata[col][c]
            atb[r] -= factor * atb[col]

    x = [0.0] * n_cols
    for col in range(n_cols - 1, -1, -1):
        s = atb[col] - sum(ata[col][c] * x[c] for c in range(col + 1, n_cols))
        x[col] = s / ata[col][col]
    return x

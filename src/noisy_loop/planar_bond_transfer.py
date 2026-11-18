import numpy as np


def transfer_log_loop(model, weights):
    rows, cols = model.rows, model.cols
    w = np.asarray(weights, float)
    if (
        w.shape != (model.dimension,)
        or not np.isfinite(w).all()
        or (np.abs(w) >= 1).any()
        or cols > 24
    ):
        raise ValueError("Require valid weights and at most 24 columns")
    state = np.arange(2**cols, dtype=np.uint64)
    flip = [
        (((state >> (cols - c - 1)) ^ (state >> (cols - c - 2))) & 1).astype(bool)
        for c in range(cols - 1)
    ]
    horizontal = w[: rows * (cols - 1)].reshape(rows, cols - 1)
    vertical = w[rows * (cols - 1) :].reshape(rows - 1, cols)
    probability = np.full(2**cols, 2.0 ** (-cols))
    result = 0.0
    for r in range(rows):
        if r:
            for c in range(cols):
                view = probability.reshape(-1, 2, 2 ** (cols - c - 1))
                a, b = view[:, 0].copy(), view[:, 1].copy()
                same, different = (
                    (1 + vertical[r - 1, c]) / 2,
                    (1 - vertical[r - 1, c]) / 2,
                )
                view[:, 0] = same * a + different * b
                view[:, 1] = different * a + same * b
        for c in range(cols - 1):
            probability *= np.where(flip[c], 1 - horizontal[r, c], 1 + horizontal[r, c])
        normalizer = probability.sum()
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise FloatingPointError("Positive transfer normalization failed")
        result += np.log(normalizer)
        probability /= normalizer
    return float(result)

import numpy as np
from scipy.linalg import solve_banded


def adjacent_faces(rows, cols):
    horizontal = rows * (cols - 1)
    faces = [
        np.array(
            [
                r * (cols - 1) + c,
                (r + 1) * (cols - 1) + c,
                horizontal + r * cols + c,
                horizontal + r * cols + c + 1,
            ]
        )
        for r in range(rows - 1)
        for c in range(cols - 1)
    ]
    lookup = {}
    for face in faces:
        for e in face:
            lookup.setdefault(int(e), []).append(face)
    return {e: pair for e, pair in lookup.items() if len(pair) == 2}


def moment_scale(c, d, rectangle, kappa):
    if not 0 < kappa < 1 or not np.isfinite([c, d, rectangle]).all():
        raise ValueError("Require finite moments and kappa in (0,1)")
    c, d = np.clip([c, d], kappa**4, 1)
    rectangle = np.clip(rectangle, kappa**6, 1)
    return float(np.clip(np.sqrt(c * rectangle / d), kappa**3, 1))


def regression_nodes(u, response, cells):
    if type(cells) is not int or cells < 1:
        raise ValueError("Require a positive integer cell count")
    u, y = np.asarray(u, float), np.asarray(response, float)
    if (
        u.ndim != 1
        or u.shape != y.shape
        or not len(u)
        or not np.isfinite(u + y).all()
        or ((u < 0) | (u > 1)).any()
        or (np.abs(y) > 1).any()
    ):
        raise ValueError("Require cube values and bounded scalar responses")
    v = u * cells
    ix = np.minimum(v.astype(int), cells - 1)
    t = v - ix
    rhs = (
        np.bincount(ix, y * (1 - t), minlength=cells + 1)
        + np.bincount(ix + 1, y * t, minlength=cells + 1)
    ) / len(u)
    band = np.zeros((3, cells + 1))
    band[0, 1:] = band[2, :-1] = 1 / (6 * cells)
    band[1] = 2 / (3 * cells)
    band[1, [0, -1]] = 1 / (3 * cells)
    return solve_banded((1, 1), band, rhs) - y.mean()


def clipped_integral(knots, values, bound):
    lo, hi = np.minimum(values[:-1], values[1:]), np.maximum(values[:-1], values[1:])
    width = hi - lo
    bottom = -bound * np.maximum(np.minimum(hi, -bound) - lo, 0)
    top = bound * np.maximum(hi - np.maximum(lo, bound), 0)
    a, b = np.maximum(lo, -bound), np.minimum(hi, bound)
    middle = np.where(b > a, (b * b - a * a) / 2, 0)
    mean = np.divide(
        bottom + middle + top,
        width,
        out=np.clip(lo, -bound, bound).copy(),
        where=width > 1e-14,
    )
    return float(np.diff(knots) @ mean)


def project_channel(knots, values, bound):
    knots, values = np.asarray(knots, float), np.asarray(values, float)
    if (
        knots.ndim != 1
        or knots.shape != values.shape
        or len(knots) < 2
        or knots[0] != 0
        or knots[-1] != 1
        or (np.diff(knots) <= 0).any()
        or not np.isfinite(knots + values).all()
        or not 0 < bound < 1
    ):
        raise ValueError("Require finite ordered cube knots and bound in (0,1)")
    lo, hi = values.min() - bound, values.max() + bound
    for _ in range(60):
        shift = (lo + hi) / 2
        if clipped_integral(knots, values - shift, bound) > 0:
            lo = shift
        else:
            hi = shift
    shifted = values - (lo + hi) / 2
    extra = []
    for i, (a, b) in enumerate(zip(shifted[:-1], shifted[1:])):
        if a == b:
            continue
        for target in [-bound, bound]:
            t = (target - a) / (b - a)
            if 0 < t < 1:
                extra.append(knots[i] + t * (knots[i + 1] - knots[i]))
    new = np.unique(np.r_[knots, extra])
    nodal = np.clip(np.interp(new, knots, shifted), -bound, bound)
    if abs(np.trapezoid(nodal, new)) > 1e-11:
        raise FloatingPointError("Channel centering failed")
    return new, nodal


def fit_channels(
    observations, rows, cols, cells, kappa=0.3, bound=0.98, known_scales=None
):
    x = np.asarray(observations, float)
    d = rows * (cols - 1) + (rows - 1) * cols
    if (
        x.ndim != 2
        or x.shape[1] != d
        or len(x) < 4
        or not np.isfinite(x).all()
        or ((x < 0) | (x > 1)).any()
    ):
        raise ValueError("Require complete cube observations with at least four rows")
    split = len(x) // 2
    sign = np.where(x > 0.5, 1.0, -1.0)
    channels, diagnostics = {}, []
    for edge, (c, dface) in adjacent_faces(rows, cols).items():
        rectangle = np.setxor1d(c, dface)
        moments = [
            float(np.prod(sign[:split, f], 1).mean()) for f in [c, dface, rectangle]
        ]
        scale = (
            moment_scale(*moments, kappa)
            if known_scales is None
            else known_scales[edge]
        )
        response = np.prod(sign[split:, c[c != edge]], 1)
        nodal = regression_nodes(x[split:, edge], response, cells) / scale
        channels[edge] = project_channel(np.linspace(0, 1, cells + 1), nodal, bound)
        diagnostics.append(
            dict(
                edge=edge,
                moments=moments,
                scale=scale,
                moment_clipped=bool(
                    moments[0] < kappa**4
                    or moments[1] < kappa**4
                    or moments[2] < kappa**6
                ),
            )
        )
    return channels, diagnostics


def squared_error(first, second):
    x = np.unique(np.r_[first[0], second[0]])
    v = np.interp(x, *first) - np.interp(x, *second)
    return float(np.diff(x) @ ((v[:-1] ** 2 + v[:-1] * v[1:] + v[1:] ** 2) / 3))

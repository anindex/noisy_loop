from collections import deque

import numpy as np

from noisy_loop.planar_bond_spline import grid
from noisy_loop.planar_channel_learning import regression_nodes


def _grid_blocks(side):
    if type(side) is not int or side < 4:
        raise ValueError("Require side >= 4")
    _, edges = grid(side, side)
    incident = [[] for _ in range(side**2)]
    for e, (u, v) in enumerate(edges):
        incident[u].append((v, e))
        incident[v].append((u, e))
    corners = [0, side - 1, side * (side - 1), side**2 - 1]
    pairs = [tuple(e for _, e in incident[c]) for c in corners]
    paired = set(sum((list(p) for p in pairs), []))
    blocks = [(e,) for e in range(len(edges)) if e not in paired] + pairs
    horizontal = side * (side - 1)
    faces = [
        set(
            [
                r * (side - 1) + c,
                (r + 1) * (side - 1) + c,
                horizontal + r * side + c,
                horizontal + r * side + c + 1,
            ]
        )
        for r in range(side - 1)
        for c in range(side - 1)
    ]
    edge_faces = [[] for _ in edges]
    for face in faces:
        for e in face:
            edge_faces[e].append(face)
    return edges, incident, blocks, faces, edge_faces


def witnesses(side):
    edges, incident, blocks, faces, _ = _grid_blocks(side)
    result = []
    for block in blocks:
        c = next(f for f in faces if set(block) <= f)
        ends = [
            v
            for v in set(edges[list(block)].ravel())
            if np.count_nonzero(edges[list(block)] == v) == 1
        ]
        start, finish = ends
        queue = deque([start])
        previous = {start: None}
        while queue and finish not in previous:
            u = queue.popleft()
            for v, e in incident[u]:
                if e not in c and v not in previous:
                    previous[v] = (u, e)
                    queue.append(v)
        if finish not in previous:
            raise ValueError("No independent cycle witness")
        d = set(block)
        v = finish
        while v != start:
            v, e = previous[v]
            d.add(e)
        q = c ^ d
        assert c & d == set(block) and max(map(len, [c, d, q])) <= 12
        result.append(
            (block, np.array(sorted(c)), np.array(sorted(d)), np.array(sorted(q)))
        )
    return result


def short_witnesses(side):
    _, _, blocks, _, edge_faces = _grid_blocks(side)
    rows = []
    for block in blocks:
        face = edge_faces[block[0]][0]
        if len(block) == 1 and len(edge_faces[block[0]]) == 2:
            other = edge_faces[block[0]][1]
            reliability_edges = ()
            cycles = [face, other, face ^ other]
        else:
            reliability_edges = tuple(sorted(face - set(block)))
            cycles = [face]
            for e in reliability_edges:
                assert len(edge_faces[e]) == 2
                other = next(f for f in edge_faces[e] if f != face)
                cycles.extend([other, face ^ other])
        rows.append(
            (
                block,
                np.array(sorted(face)),
                reliability_edges,
                [tuple(sorted(f)) for f in cycles],
            )
        )
    return rows


def _short_calibration(sign, side, kappa, bound):
    moments, reliabilities = {}, {}

    def moment(cycle):
        if cycle not in moments:
            raw = float(np.prod(sign[:, cycle], axis=1).mean())
            moments[cycle] = (raw, float(np.clip(raw, kappa ** len(cycle), 1)))
        return moments[cycle][1]

    rows = []
    for block, face, interior, cycles in short_witnesses(side):
        values = [moment(c) for c in cycles]
        reliability_clipped = 0
        if not interior:
            scale = float(
                np.clip(np.sqrt(values[0] * values[2] / values[1]), kappa**3, 1)
            )
        else:
            for j, e in enumerate(interior):
                if e not in reliabilities:
                    raw = float(
                        np.sqrt(values[0] * values[2 * j + 1] / values[2 * j + 2])
                    )
                    reliabilities[e] = (raw, float(np.clip(raw, kappa, bound)))
                reliability_clipped += reliabilities[e][0] != reliabilities[e][1]
            scale = float(np.prod([reliabilities[e][1] for e in interior]))
        rows.append(
            (
                block,
                face,
                scale,
                values[0],
                dict(
                    lengths=list(map(len, cycles)),
                    clipped=sum(moments[c][0] != moments[c][1] for c in cycles),
                    calibration="short",
                    moment_cycles=[list(c) for c in cycles],
                    raw_moments=[moments[c][0] for c in cycles],
                    reliability_clipped=int(reliability_clipped),
                ),
            )
        )
    return rows


def masses(cells):
    w = np.ones(cells + 1) / cells
    w[[0, -1]] /= 2
    return w


def project_nodes(
    values, bound, *, return_diagnostics=False, gap_tolerance=1e-9, max_iterations=20000
):
    v = np.asarray(values, float)
    if (
        v.ndim not in [1, 2]
        or min(v.shape) < 2
        or not np.isfinite(v).all()
        or (v.ndim == 2 and v.shape[0] != v.shape[1])
        or not 0 < bound < 1
        or not np.isfinite(gap_tolerance)
        or gap_tolerance <= 0
        or type(max_iterations) is not int
        or max_iterations < 1
    ):
        raise ValueError("Require finite scalar/tensor nodes and bound in (0,1)")
    w = masses(v.shape[0] - 1)
    weight = w if v.ndim == 1 else np.outer(w, w)

    def centered(z):
        if z.ndim == 1:
            return z - w @ z
        return z - (z @ w)[:, None] - (w @ z)[None, :] + w @ z @ w

    def certificate(x, p, q, iterations):
        x = centered(x)
        x *= min(1.0, bound / max(bound, np.max(np.abs(x))))
        q = q - centered(q)
        residual = x - (v - p - q)

        complementarity = np.sum(weight * np.abs(p) * (bound - np.sign(p) * x))
        quadratic = 0.5 * np.sum(weight * residual**2)
        orthogonality = abs(np.sum(weight * q * x))
        allowance = (
            8
            * np.finfo(float).eps
            * (1 + quadratic + bound * np.sum(weight * np.abs(p)))
        )
        gap = float(quadratic + complementarity + orthogonality + allowance)
        centering = max(float(np.max(np.abs(x @ w))), float(np.max(np.abs(w @ x))))
        amplitude = max(0.0, float(np.max(np.abs(x)) - bound))
        info = dict(
            iterations=iterations,
            objective=float(0.5 * np.sum(weight * (x - v) ** 2)),
            gap=gap,
            squared_distance_bound=2 * gap,
            centering_residual=centering,
            amplitude_violation=amplitude,
            roundoff_allowance=float(allowance),
            orthogonality_residual=float(orthogonality),
            gap_tolerance=gap_tolerance,
        )
        return x, info

    if v.ndim == 1:
        lo, hi = v.min() - bound, v.max() + bound
        for _ in range(60):
            shift = (lo + hi) / 2
            if w @ np.clip(v - shift, -bound, bound) > 0:
                lo = shift
            else:
                hi = shift
        q = np.full_like(v, (lo + hi) / 2)
        x = np.clip(v - q, -bound, bound)
        x, info = certificate(x, v - q - x, q, 60)
        if info["gap"] > gap_tolerance or info["centering_residual"] > 1e-12:
            raise FloatingPointError(f"Scalar projection tolerance unmet: {info}")
        return (x, info) if return_diagnostics else x

    x, p, q = v.copy(), np.zeros_like(v), np.zeros_like(v)
    for iteration in range(1, max_iterations + 1):
        y = np.clip(x + p, -bound, bound)
        p = x + p - y
        z = y + q
        new = centered(z)
        q = z - new
        x = new
        feasible, info = certificate(x, p, q, iteration)
        if info["gap"] <= gap_tolerance and info["centering_residual"] <= 1e-12:
            return (feasible, info) if return_diagnostics else feasible
    raise FloatingPointError(f"Tensor projection tolerance unmet: {info}")


def fit_blocks(
    x, side, cells, kappa=0.3, bound=0.98, *, calibration="short", sample_split=False
):
    """Fit scalar channels and corner products from complete observation vectors.

    x has shape (N, 2*side*(side-1)), in grid edge order, with known uniform
    margins. kappa is an input reliability lower bound; bound is an amplitude
    cap. Return a dictionary keyed by edge tuples and a list of diagnostics.
    Each scalar has cells+1 knot values; each corner has a square knot array.
    """
    x = np.asarray(x, float)
    if (
        x.ndim != 2
        or x.shape[1] != 2 * side * (side - 1)
        or len(x) < 4
        or not np.isfinite(x).all()
        or ((x < 0) | (x > 1)).any()
        or not 0 < kappa < bound < 1
        or calibration not in ["legacy", "short"]
        or type(sample_split) is not bool
    ):
        raise ValueError("Require cube observations and 0 < kappa < bound < 1")
    split = len(x) // 2
    sign = np.where(x > 0.5, 1.0, -1.0)
    calibration_sign = sign[:split] if sample_split else sign
    regression_x = x[split:] if sample_split else x
    regression_sign = sign[split:] if sample_split else sign
    fitted, diagnostics = {}, []
    if calibration == "short":
        rows = _short_calibration(calibration_sign, side, kappa, bound)
    else:
        rows = []
        for block, c, d, q in witnesses(side):
            raw = [
                float(np.prod(calibration_sign[:, cycle], 1).mean())
                for cycle in [c, d, q]
            ]
            moment = [
                float(np.clip(m, kappa ** len(cycle), 1))
                for m, cycle in zip(raw, [c, d, q])
            ]
            alpha = float(
                np.clip(
                    np.sqrt(moment[0] * moment[2] / moment[1]),
                    kappa ** (len(c) - len(block)),
                    1,
                )
            )
            rows.append(
                (
                    block,
                    c,
                    alpha,
                    moment[0],
                    dict(
                        lengths=list(map(len, [c, d, q])),
                        clipped=sum(m != r for m, r in zip(moment, raw)),
                    ),
                )
            )
    for block, c, alpha, face_moment, diagnostic in rows:
        regressions = [
            regression_nodes(
                regression_x[:, e], np.prod(regression_sign[:, c[c != e]], 1), cells
            )
            for e in block
        ]
        if len(block) == 1:
            fitted[block], projection = project_nodes(
                regressions[0] / alpha, bound, return_diagnostics=True
            )
            preliminary = []
            estimator_error = np.sqrt(2 * projection["gap"])
        else:
            regression_bound = max(1 - 1e-12, bound**4)
            projected = [
                project_nodes(r, regression_bound, return_diagnostics=True)
                for r in regressions
            ]
            g, preliminary = zip(*projected)
            fitted[block], projection = project_nodes(
                np.outer(*g) / (alpha * face_moment), bound**2, return_diagnostics=True
            )
            estimator_error = np.sqrt(2 * projection["gap"]) + sum(
                np.sqrt(2 * p["gap"]) for p in preliminary
            ) / (alpha * face_moment)
        diagnostics.append(
            dict(
                block=list(block),
                **diagnostic,
                scale=alpha,
                projection=projection,
                preliminary_projections=list(preliminary),
                squared_estimator_error_bound=float(estimator_error**2),
            )
        )
    return fitted, diagnostics


def evaluate_nodes(nodes, x):
    cells = len(nodes) - 1
    values = np.asarray(x, float)
    p = values * cells
    ix = np.minimum(p.astype(int), cells - 1)
    frac = p - ix
    if nodes.ndim == 1:
        return nodes[ix] * (1 - frac) + nodes[ix + 1] * frac
    return sum(
        nodes[ix[:, 0] + a, ix[:, 1] + b]
        * (frac[:, 0] if a else 1 - frac[:, 0])
        * (frac[:, 1] if b else 1 - frac[:, 1])
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
    )


def grid_weights(fitted, x):
    out = np.empty_like(x)
    for block, nodes in fitted.items():
        if len(block) == 1:
            out[:, block[0]] = evaluate_nodes(nodes, x[:, block[0]])
        else:
            v = evaluate_nodes(nodes, x[:, block])
            root = np.sqrt(np.abs(v))
            out[:, block[0]], out[:, block[1]] = root, np.sign(v) * root
    return out


def evidence_basis(lower, upper, cells):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (
        type(cells) is not int
        or cells < 1
        or lo.shape != hi.shape
        or lo.ndim != 1
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or (lo < 0).any()
        or (hi > 1).any()
        or (hi < lo).any()
    ):
        raise ValueError("Require finite ordered cube bounds and positive cell count")
    left = np.arange(cells) / cells
    a = np.maximum(lo[:, None], left)
    b = np.minimum(hi[:, None], left + 1 / cells)
    width = np.maximum(b - a, 0)
    fraction = ((a - left) + (b - left)) * (cells / 2)
    basis = np.zeros((len(lo), cells + 1))
    basis[:, :-1] += width * (1 - fraction)
    basis[:, 1:] += width * fraction
    point = np.flatnonzero(lo == hi)
    index = np.minimum((lo[point] * cells).astype(int), cells - 1)
    fraction = lo[point] * cells - index
    basis[point, index] = 1 - fraction
    basis[point, index + 1] = fraction
    return basis


def block_evidence(nodes, lower, upper):
    v = np.asarray(nodes, float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (
        v.ndim not in [1, 2]
        or min(v.shape) < 2
        or (v.ndim == 2 and v.shape[0] != v.shape[1])
        or not np.isfinite(v).all()
        or (np.abs(v) >= 1).any()
        or lo.ndim != 2
        or lo.shape != hi.shape
        or lo.shape[1] != v.ndim
    ):
        raise ValueError(
            "Require bounded scalar/tensor nodes and matching evidence bounds"
        )
    w = masses(len(v) - 1)
    if np.max(np.abs(v @ w)) > 1e-9 or (v.ndim == 2 and np.max(np.abs(w @ v)) > 1e-9):
        raise ValueError("Require separately centered channel nodes")
    basis = [evidence_basis(lo[:, j], hi[:, j], len(v) - 1) for j in range(v.ndim)]
    alpha = np.prod(np.where(lo == hi, 1.0, hi - lo), axis=1)
    beta = (
        basis[0] @ v
        if v.ndim == 1
        else np.einsum("ni,ij,nj->n", *[basis[0], v, basis[1]])
    )
    if (
        (alpha <= 0).any()
        or not np.isfinite(beta).all()
        or (np.abs(beta) >= alpha).any()
    ):
        raise FloatingPointError("Local evidence underflowed or lost strict positivity")
    return alpha, beta


def log_block_evidence(model, fitted, lower, upper):
    """Return one log evidence likelihood per row for an independent-spin fit.

    Bounds have shape (N, model.dimension). Equal bounds specify a point
    density; [0, 1] integrates out a reading. Other bounds specify intervals.
    """
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (
        model.rows != model.cols
        or model.prior.any()
        or lo.ndim != 2
        or lo.shape != hi.shape
        or lo.shape[1] != model.dimension
        or set(fitted) != {b for b, *_ in witnesses(model.rows)}
    ):
        raise ValueError(
            "Require a complete fitted zero-prior square grid and matching bounds"
        )
    weights = np.empty_like(lo)
    log_mass = np.zeros(len(lo))
    for block, nodes in fitted.items():
        alpha, beta = block_evidence(nodes, lo[:, block], hi[:, block])
        log_mass += np.log(alpha)
        value = beta / alpha
        if len(block) == 1:
            weights[:, block[0]] = value
        else:
            root = np.sqrt(np.abs(value))
            weights[:, block[0]], weights[:, block[1]] = root, np.sign(value) * root
    return log_mass + np.array([model.log_loop(w) for w in weights])

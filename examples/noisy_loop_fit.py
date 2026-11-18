import numpy as np

from noisy_loop.noisy_loop_calibration import curve, sample_loop
from noisy_loop.observable_blocks import (
    block_evidence,
    fit_blocks,
    log_block_evidence,
    masses,
)
from noisy_loop.planar_bond_spline import PlanarBondSpline
from noisy_loop.planar_bond_transfer import transfer_log_loop


def main():
    side, cells = 4, 8
    graph = PlanarBondSpline(side, side)
    train = sample_loop(side, "ramp", 0.83125, 8192, 20261102)
    test = sample_loop(side, "ramp", 0.83125, 1024, 20261103)
    nodes, diagnostics = fit_blocks(
        train,
        side,
        cells,
        kappa=0.2,
        bound=0.98,
        calibration="short",
        sample_split=False,
    )
    w = masses(cells)
    for block, values in nodes.items():
        assert np.max(np.abs(values)) <= 0.98 ** len(block) + 1e-12
        np.testing.assert_allclose(values @ w, 0, atol=1e-12)
        if values.ndim == 2:
            np.testing.assert_allclose(w @ values, 0, atol=1e-12)
    zero = np.zeros((1, graph.dimension))
    np.testing.assert_allclose(
        log_block_evidence(graph, nodes, zero, zero + 1), 0, atol=1e-12
    )
    point = log_block_evidence(graph, nodes, test, test)
    lo, hi = test[:8].copy(), test[:8].copy()
    lo[:, ::2], hi[:, ::2] = 0, 1
    missing = log_block_evidence(graph, nodes, lo, hi)
    conditional = point[:8] - missing
    interval_lo, interval_hi = (
        np.maximum(0, test[:8] - 0.05),
        np.minimum(1, test[:8] + 0.05),
    )
    interval = log_block_evidence(graph, nodes, interval_lo, interval_hi)
    assert all(np.isfinite(x).all() for x in [point, missing, conditional, interval])
    assert np.max(interval) <= 1e-12
    truth = np.array([graph.log_loop(row) for row in curve(test, "ramp", 0.83125)])
    errors = []
    for lower, upper, actual in [
        (test[:8], test[:8], point[:8]),
        (lo, hi, missing),
        (interval_lo, interval_hi, interval),
    ]:
        weights, log_mass = np.empty_like(lower), np.zeros(len(lower))
        for block, values in nodes.items():
            alpha, beta = block_evidence(values, lower[:, block], upper[:, block])
            log_mass += np.log(alpha)
            value = beta / alpha
            if len(block) == 1:
                weights[:, block[0]] = value
            else:
                root = np.sqrt(np.abs(value))
                weights[:, block[0]], weights[:, block[1]] = root, np.sign(value) * root
        reference = log_mass + np.array(
            [transfer_log_loop(graph, row) for row in weights]
        )
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-11)
        errors.extend(np.abs(actual - reference))
    print(
        f"Full-sample fit: {len(nodes)} observable blocks, {len(train)} training vectors."
    )
    print(
        "Centering, amplitude, normalization, point, missing and interval checks passed."
    )
    print(
        f"Held-out Monte Carlo KL (nats): fitted={(truth - point).mean():.6f}, "
        f"independence={truth.mean():.6f}; paired excess={-point.mean():.6f}."
    )
    print(
        f"Maximum projection gap: {max(d['projection']['gap'] for d in diagnostics):.3e}."
    )
    print(f"Maximum log-evidence discrepancy over 24 queries: {max(errors):.3e}.")
    print(
        f"First conditional log density of even-indexed readings given the others: {conditional[0]:.6f}."
    )
    print(
        "Transfer independently checks global contraction; local spline integrals are shared."
    )


if __name__ == "__main__":
    main()

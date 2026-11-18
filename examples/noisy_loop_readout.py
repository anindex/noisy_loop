from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from noisy_loop.planar_bond_spline import PlanarBondSpline


def main():
    path = Path(__file__).parent / "data/readout_iq.npz"
    with np.load(path, allow_pickle=False) as data:
        pools = [
            data[f"cal_q3_first{y}"] * data[f"cal_q3_first{y}_scale"] for y in (0, 1)
        ]
    assert all(x.shape == (15748, 2) and np.isfinite(x).all() for x in pools)
    calibration = np.concatenate([x[:8000] for x in pools])
    center, scale = calibration.mean(0), calibration.std(0)
    assert (scale > 0).all()
    models = []
    for pool in pools:
        x = (pool[:8000] - center) / scale
        models.append(multivariate_normal(x.mean(0), np.cov(x.T) + 1e-6 * np.eye(2)))

    graph = PlanarBondSpline(2, 2)
    rng = np.random.default_rng(20261118)
    spins = rng.choice([-1, 1], size=(8192, 4))
    bonds = spins[:, graph.edges[:, 0]] * spins[:, graph.edges[:, 1]]
    labels = (bonds + 1) // 2
    iq = np.empty((*labels.shape, 2))
    for y, pool in enumerate(pools):
        mask = labels == y
        iq[mask] = pool[rng.integers(8000, len(pool), size=mask.sum())]
    z = (iq - center) / scale
    log_ratio = models[1].logpdf(z) - models[0].logpdf(z)
    weights = 0.96 * np.tanh(log_ratio / 2)
    other = np.prod(weights[:, 1:], axis=1)
    loop_mean = (weights[:, 0] + other) / (1 + np.prod(weights, axis=1))

    states = np.array(list(product([-1, 1], repeat=4)))
    all_bonds = states[:, graph.edges[:, 0]] * states[:, graph.edges[:, 1]]
    for row, expected_mean in zip(weights[:32], loop_mean[:32]):
        likelihood = np.prod(1 + all_bonds * row, axis=1)
        log_z, gradient = graph.log_loop(row, gradient=True)
        np.testing.assert_allclose(np.exp(log_z), likelihood.mean(), atol=1e-12)
        np.testing.assert_allclose(
            expected_mean, likelihood @ all_bonds[:, 0] / likelihood.sum(), atol=1e-12
        )
        np.testing.assert_allclose(
            expected_mean, row[0] + (1 - row[0] ** 2) * gradient[0], atol=1e-12
        )
        forest = row.copy()
        forest[0] = 0
        np.testing.assert_allclose(graph.log_loop(forest), 0, atol=1e-12)

    print(
        "Calibration: 8,000 measured IQ pairs per class; Gaussian channels with 2% class mixing."
    )
    print(
        "Evaluation: 8,192 synthetic loops, resampled from the remaining 7,748 pairs per class."
    )
    print("Predict the first bond; lower log loss is better (nats per bond).")
    for name, mean in [
        ("No readings", np.zeros(len(labels))),
        ("Other three readings", other),
        ("First reading alone", weights[:, 0]),
        ("All four readings", loop_mean),
    ]:
        probability = (1 + mean) / 2
        loss = -np.mean(
            np.where(labels[:, 0], np.log(probability), np.log1p(-probability))
        )
        accuracy = np.mean((probability >= 0.5) == labels[:, 0])
        print(f"{name:24s} log loss={loss:.6f}, accuracy={accuracy:.4f}")
    print(
        "32 determinant and posterior checks agree with enumeration over 16 spin states."
    )
    print(
        "Measured noise on a synthetic graph; resampling does not create new hardware trials."
    )


if __name__ == "__main__":
    main()

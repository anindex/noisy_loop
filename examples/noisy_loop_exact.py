import itertools

import numpy as np

from noisy_loop.planar_bond_spline import PlanarBondSpline


def main():
    graph = PlanarBondSpline(3, 3, amplitude=0.6, ramp=0.25)
    rng = np.random.default_rng(392)
    weights = rng.uniform(-0.6, 0.6, graph.dimension)
    spins = np.array(list(itertools.product([-1.0, 1.0], repeat=9)))
    bonds = spins[:, graph.edges[:, 0]] * spins[:, graph.edges[:, 1]]
    direct = np.prod(1 + bonds * weights, axis=1).mean()
    determinant = np.exp(graph.log_loop(weights))
    np.testing.assert_allclose(determinant, direct, rtol=1e-12)
    forest = weights.copy()
    forest[6:] = 0
    np.testing.assert_allclose(
        np.prod(1 + bonds * forest, axis=1).mean(), 1, atol=1e-12
    )
    np.testing.assert_allclose(np.exp(graph.log_loop(np.zeros(12))), 1, atol=1e-12)
    print(
        f"512 spin states, 12 observations: exact={direct:.12f}, determinant={determinant:.12f}"
    )
    print("Normalization and forest independence checks passed.")


if __name__ == "__main__":
    main()

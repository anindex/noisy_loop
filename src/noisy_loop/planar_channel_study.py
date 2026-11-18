import numpy as np

from noisy_loop.planar_bond_spline import emission_quantile


def truth_for(graph, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.72, 0.95, graph.dimension)
    ramp = rng.uniform(0.1, 0.75, graph.dimension)
    curves = [
        (np.array([0, (1 - t) / 2, (1 + t) / 2, 1.0]), np.array([-s, -s, s, s]))
        for s, t in zip(a, ramp)
    ]
    return a, ramp, curves


def sample(graph, amplitude, ramp, n, seed):
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1.0, 1.0], size=(n, len(graph.xy)))
    bonds = spins[:, graph.edges[:, 0]] * spins[:, graph.edges[:, 1]]
    u = rng.random(bonds.shape)
    for e in range(graph.dimension):
        u[:, e] = emission_quantile(u[:, e], bonds[:, e], amplitude[e], ramp[e])
    return u, bonds

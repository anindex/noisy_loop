import numpy as np
from scipy.optimize import brentq

from noisy_loop.planar_bond_spline import PlanarBondSpline

SCALE = brentq(lambda s: s * (np.logaddexp(1 / s, -1 / s) - np.log(2)) - 0.875, 0.01, 1)


def curve(u, target, eta):
    a = eta / 0.875
    return a * (
        np.tanh((2 * u - 1) / SCALE)
        if target == "smooth"
        else np.clip((2 * u - 1) / 0.25, -1, 1)
    )


def sample_loop(side, target, eta, n, seed):
    """Draw n independent vectors on a square grid with side vertices per row.

    The ramp and smooth targets have common edge reliability eta. The result
    has shape (n, 2*side*(side-1)), with columns ordered as PlanarBondSpline.edges.
    """
    if target not in ("ramp", "smooth") or type(n) is not int or n < 1:
        raise ValueError(
            "Require ramp or smooth target and a positive integer sample count"
        )
    graph = PlanarBondSpline(side, side, amplitude=eta / 0.875, ramp=0.25)
    rng = np.random.default_rng(seed)
    out = np.empty((n, graph.dimension))
    for first in range(0, n, 4096):
        size = min(4096, n - first)
        if target == "ramp":
            out[first : first + size] = graph.sample(size, int(rng.integers(2**32)))
        else:
            spins = rng.choice([-1.0, 1.0], (size, side**2))
            bonds = spins[:, graph.edges[:, 0]] * spins[:, graph.edges[:, 1]]
            for e in range(graph.dimension):
                pending = np.arange(size)
                while len(pending):
                    u = rng.random(len(pending))
                    accept = rng.random(len(pending)) * (1 + eta / 0.875) <= 1 + bonds[
                        pending, e
                    ] * curve(u, target, eta)
                    out[first + pending[accept], e] = u[accept]
                    pending = pending[~accept]
    assert ((out > 0) & (out < 1)).all()
    return out


def integration(target, eta, cells=8, degree=24):
    knots = np.unique(np.r_[np.linspace(0, 1, cells + 1), 0.375, 0.625])
    p, w = np.polynomial.legendre.leggauss(degree)
    x = (
        (knots[:-1, None] + knots[1:, None]) / 2 + np.diff(knots)[:, None] * p / 2
    ).ravel()
    w = (np.diff(knots)[:, None] * w / 2).ravel()
    position = x * cells
    ix = np.minimum(position.astype(int), cells - 1)
    fraction = position - ix
    hats = np.zeros((len(x), cells + 1))
    hats[np.arange(len(x)), ix] = 1 - fraction
    hats[np.arange(len(x)), ix + 1] = fraction
    h = curve(x, target, eta)
    return hats.T @ (w[:, None] * hats), hats.T @ (w * h), float(w @ (h * h))


def block_error(nodes, mass, cross, energy):
    if nodes.ndim == 1:
        return float(nodes @ mass @ nodes - 2 * nodes @ cross + energy)
    return float(
        np.sum(nodes * (mass @ nodes @ mass)) - 2 * cross @ nodes @ cross + energy**2
    )

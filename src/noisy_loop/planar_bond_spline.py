import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu

from noisy_loop.spline_memory_information import _parameters, channel


def grid(rows, cols):
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 2:
        raise ValueError("Require at least two rows and columns")
    xy = np.array([(c, r) for r in range(rows) for c in range(cols)])
    horizontal = [
        (r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)
    ]
    vertical = [
        (r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)
    ]
    return xy, np.array(horizontal + vertical, dtype=int)


def channel_integral(u, amplitude, ramp):
    _parameters(0.0, amplitude, ramp)
    u = np.asarray(u, float)
    if not np.isfinite(u).all() or ((u < 0) | (u > 1)).any():
        raise ValueError("Require finite cube values")
    left, right = (1 - ramp) / 2, (1 + ramp) / 2
    return amplitude * (
        -u + np.maximum(u - left, 0) ** 2 / ramp - np.maximum(u - right, 0) ** 2 / ramp
    )


def emission_quantile(probability, mean, amplitude, ramp):
    _parameters(0.0, amplitude, ramp)
    p, m = np.broadcast_arrays(np.asarray(probability, float), np.asarray(mean, float))
    if (
        not np.isfinite(p).all()
        or not np.isfinite(m).all()
        or ((p < 0) | (p > 1)).any()
        or (np.abs(m) > 1).any()
    ):
        raise ValueError("Require probabilities and bond means in their domains")
    knots = np.unique([0.0, (1 - ramp) / 2, (1 + ramp) / 2, 1.0])
    values = 1 + m[..., None] * channel(knots, amplitude, ramp)
    cdf = knots + m[..., None] * channel_integral(knots, amplitude, ramp)
    idx = np.clip((p[..., None] >= cdf).sum(-1) - 1, 0, len(knots) - 2)
    take = lambda x: np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]
    residual = p - take(cdf)
    a = take(values)
    b = take(np.diff(values, axis=-1) / np.diff(knots))
    disc = a * a + 2 * b * residual
    if (disc < -1e-13).any():
        raise FloatingPointError("Invalid spline inverse discriminant")
    return np.clip(
        knots[idx] + 2 * residual / (a + np.sqrt(np.maximum(disc, 0))), 0.0, 1.0
    )


def _permutation_parity(p):
    seen = np.zeros(len(p), bool)
    cycles = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        cycles += 1
        while not seen[i]:
            seen[i] = True
            i = p[i]
    return (len(p) - cycles) % 2


class PlanarBondSpline:
    """Known ramp channels on a rectangular grid, queried by sparse LU.

    Edges are ordered horizontally, then vertically, each in row-major order.
    Optional couplings specify a fixed zero-field Ising spin prior; the default
    uses independent uniform spins. Query batches have one column per edge.
    """

    def __init__(self, rows, cols, amplitude=0.95, ramp=0.0625, couplings=None):
        _parameters(0.0, amplitude, ramp)
        self.xy, self.edges = grid(rows, cols)
        self.rows, self.cols = rows, cols
        self.dimension = len(self.edges)
        self.amplitude, self.ramp = amplitude, ramp
        j = (
            np.zeros(self.dimension)
            if couplings is None
            else np.asarray(couplings, float)
        )
        if j.shape != (self.dimension,) or not np.isfinite(j).all():
            raise ValueError("Require one finite coupling per edge")
        self.prior = np.tanh(j)
        if (np.abs(self.prior) >= 1).any():
            raise ValueError("Coupling saturates tanh in float64")
        directed = np.concatenate([self.edges, self.edges[:, ::-1]])
        direction = self.xy[directed[:, 1]] - self.xy[directed[:, 0]]
        outgoing = [[] for _ in self.xy]
        for k, (a, _) in enumerate(directed):
            outgoing[a].append(k)
        row, col, phase = [], [], []
        for i, (a, b) in enumerate(directed):
            for k in outgoing[b]:
                if directed[k, 1] == a:
                    continue
                x, y = direction[i], direction[k]
                angle = np.arctan2(x[0] * y[1] - x[1] * y[0], x @ y)
                row.append(i)
                col.append(k)
                phase.append(np.exp(0.5j * angle))
        self.turn = csc_matrix((phase, (row, col)), shape=(2 * self.dimension,) * 2)
        self.identity = eye(2 * self.dimension, dtype=complex, format="csc")
        self.log_normalizer = self.log_loop(self.prior)

    def _input(self, values):
        x = np.asarray(values, float)
        if (
            x.ndim != 2
            or x.shape[1] != self.dimension
            or np.isinf(x).any()
            or ((x < 0) | (x > 1)).any()
        ):
            raise ValueError(
                "Require a batch of cube observations; NaN denotes missing"
            )
        return x

    def _factor(self, weight):
        w = np.asarray(weight, float)
        if (
            w.shape != (self.dimension,)
            or not np.isfinite(w).all()
            or (np.abs(w) >= 1).any()
        ):
            raise ValueError("Require one weight strictly inside (-1,1) per edge")
        matrix = self.identity - self.turn.multiply(np.tile(w, 2)).tocsc()
        matrix.eliminate_zeros()
        return splu(matrix)

    def log_loop(self, weight, gradient=False):
        """Return log L_G(weight), optionally with its gradient in edge weights."""
        factor = self._factor(weight)
        diagonal = factor.U.diagonal()
        phase = np.angle(diagonal).sum() + np.pi * (
            _permutation_parity(factor.perm_r) + _permutation_parity(factor.perm_c)
        )
        if not np.isfinite(diagonal).all() or abs(np.exp(1j * phase) - 1) > 1e-7:
            raise FloatingPointError(
                "Kac--Ward determinant lost its positive-real phase"
            )
        value = float(0.5 * np.log(np.abs(diagonal)).sum())
        if not gradient:
            return value

        directed = np.empty(2 * self.dimension)
        for start in range(0, len(directed), 32):
            stop = min(start + 32, len(directed))
            solved = factor.solve(self.turn[:, start:stop].toarray())
            directed[start:stop] = (
                -0.5 * solved[np.arange(start, stop), np.arange(stop - start)].real
            )
        return value, directed[: self.dimension] + directed[self.dimension :]

    def log_evidence(self, values):
        """Return marginal log densities, using NaN for missing readings."""
        h = channel(self._input(values), self.amplitude, self.ramp)
        weight = (self.prior + h) / (1 + self.prior * h)
        return (
            np.array([self.log_loop(w) for w in weight])
            + np.log1p(self.prior * h).sum(-1)
            - self.log_normalizer
        )

    def log_rectangle(self, lower, upper):
        """Return log probabilities for positive-width intervals in every column."""
        lo, hi = self._input(lower), self._input(upper)
        if (
            lo.shape != hi.shape
            or not np.isfinite(lo).all()
            or not np.isfinite(hi).all()
            or (hi <= lo).any()
        ):
            raise ValueError("Require positive-width finite intervals")
        h = (
            channel_integral(hi, self.amplitude, self.ramp)
            - channel_integral(lo, self.amplitude, self.ramp)
        ) / (hi - lo)
        weight = (self.prior + h) / (1 + self.prior * h)
        return (
            np.array([self.log_loop(w) for w in weight])
            + np.log((hi - lo) * (1 + self.prior * h)).sum(-1)
            - self.log_normalizer
        )

    def posterior_bonds(self, values):
        """Return posterior bond means; NaN marks missing readings."""
        h = channel(self._input(values), self.amplitude, self.ramp)
        weight = (self.prior + h) / (1 + self.prior * h)
        gradient = np.array([self.log_loop(w, gradient=True)[1] for w in weight])
        mean = weight + (1 - weight * weight) * gradient
        if (np.abs(mean) > 1 + 1e-9).any():
            raise FloatingPointError("Posterior bond mean outside [-1,1]")
        return np.clip(mean, -1, 1)

    def conditional_log_prob(self, values):
        """Score each reading given all others in a complete observation batch."""
        x = self._input(values)
        if np.isnan(x).any():
            raise ValueError(
                "Scalar leave-one-out scores require complete observations"
            )
        h = channel(x, self.amplitude, self.ramp)
        mean = self.posterior_bonds(x)
        return np.log1p(-h * h) - np.log1p(-mean * h)

    def sample(self, n, seed, evidence=None):
        """Draw n vectors, optionally fixing one evidence vector with NaN gaps."""
        if type(n) is not int or n < 1:
            raise ValueError("Require a positive sample count")
        rng = np.random.default_rng(seed)
        if evidence is None and not self.prior.any():
            spins = 2 * rng.integers(0, 2, size=(n, len(self.xy))) - 1
            bonds = spins[:, self.edges[:, 0]] * spins[:, self.edges[:, 1]]
            return emission_quantile(
                rng.random(bonds.shape), bonds, self.amplitude, self.ramp
            )
        x = (
            np.full(self.dimension, np.nan)
            if evidence is None
            else np.asarray(evidence, float)
        )
        self._input(x[None])
        missing = np.flatnonzero(np.isnan(x))
        h = channel(x, self.amplitude, self.ramp)
        initial = (self.prior + h) / (1 + self.prior * h)
        initial_inverse = self._factor(initial).solve(
            np.eye(2 * self.dimension, dtype=complex)
        )
        samples = np.broadcast_to(x, (n, self.dimension)).copy()
        for sample in samples:
            w, inverse = initial.copy(), initial_inverse.copy()
            for e in missing:
                columns = np.array([e, e + self.dimension])

                product = inverse @ self.turn[:, columns].toarray()
                grad = -0.5 * np.trace(product[columns]).real
                mean = w[e] + (1 - w[e] ** 2) * grad
                if abs(mean) > 1 + 1e-8:
                    raise FloatingPointError("Sequential bond mean outside [-1,1]")
                sample[e] = emission_quantile(
                    rng.random(), np.clip(mean, -1, 1), self.amplitude, self.ramp
                )
                h_e = float(channel(sample[e], self.amplitude, self.ramp))
                new = (self.prior[e] + h_e) / (1 + self.prior[e] * h_e)
                delta = new - w[e]
                update = np.linalg.solve(
                    np.eye(2) - delta * product[columns], inverse[columns]
                )
                inverse += delta * product @ update
                w[e] = new
        return samples

    def spectral_certificate(self, relative_squared_error=0.01):
        if self.prior.any() or not 0 <= relative_squared_error < 1:
            raise ValueError("Require zero prior and relative squared error in [0,1)")
        v = self.amplitude**2 * (1 - 2 * self.ramp / 3)
        first = self.log_loop(np.full(self.dimension, v))
        second = self.log_loop(np.full(self.dimension, v * v))
        return dict(
            exact_log2_rank=0
            if self.amplitude == 0
            else (self.rows - 1) * (self.cols - 1),
            log_squared_L2_norm=first,
            log_fourth_spectral_sum=second,
            approximate_log2_rank_lower=max(
                0.0,
                (2 * np.log1p(-relative_squared_error) + 2 * first - second)
                / np.log(2),
            ),
        )

    def disjoint_plaquettes(self):
        lookup = {tuple(e): i for i, e in enumerate(self.edges)}
        return np.array(
            [
                [
                    lookup[(r * self.cols + c, r * self.cols + c + 1)],
                    lookup[((r + 1) * self.cols + c, (r + 1) * self.cols + c + 1)],
                    lookup[(r * self.cols + c, (r + 1) * self.cols + c)],
                    lookup[(r * self.cols + c + 1, (r + 1) * self.cols + c + 1)],
                ]
                for r in range(0, self.rows - 1, 2)
                for c in range(0, self.cols - 1, 2)
            ]
        )

    def log_overlap(self, other):
        if (
            (self.rows, self.cols) != (other.rows, other.cols)
            or self.prior.any()
            or other.prior.any()
        ):
            raise ValueError("Overlap formula requires the same grid and zero priors")
        lo, hi = sorted([self.ramp, other.ramp])
        inner = self.amplitude * other.amplitude * (1 - hi / 2 - lo * lo / (6 * hi))
        return self.log_loop(np.full(self.dimension, inner))

    def fit_amplitude(self, values, bounds=(0.1, 0.99)):
        x = self._input(values)
        lo, hi = bounds
        if len(x) == 0 or np.isnan(x).any() or self.prior.any() or not 0 < lo < hi < 1:
            raise ValueError("Require complete data, zero prior and 0 < bounds < 1")
        r = np.clip((2 * x - 1) / self.ramp, -1, 1)
        moment = np.prod(r[:, self.disjoint_plaquettes()], axis=-1).mean()
        kappa = 1 - 2 * self.ramp / 3
        return float(np.clip(moment / kappa**4, lo**4, hi**4) ** 0.25)

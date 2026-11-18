from math import ceil, comb, log

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri, xlogy


def _parameters(beta, amplitude, ramp):
    if not (
        np.isfinite(beta)
        and abs(beta) < 1
        and np.isfinite(amplitude)
        and 0 <= amplitude < 1
        and np.isfinite(ramp)
        and 0 < ramp <= 1
    ):
        raise ValueError("Require |beta| < 1, 0 <= amplitude < 1, and 0 < ramp <= 1")


def channel(values, amplitude=0.95, ramp=1.0):
    _parameters(0.0, amplitude, ramp)
    x = np.asarray(values, dtype=float)
    if np.isinf(x).any() or ((x < 0) | (x > 1)).any():
        raise ValueError("Require cube values; NaN denotes missing evidence")
    return amplitude * np.clip((2 * np.nan_to_num(x, nan=0.5) - 1) / ramp, -1, 1)


def quadrature(ramp, order=64):
    _parameters(0.0, 0.0, ramp)
    if type(order) is not int or order < 2:
        raise ValueError("Require quadrature order >= 2")
    z, w = np.polynomial.legendre.leggauss(order)
    knots = np.unique([0.0, (1 - ramp) / 2, (1 + ramp) / 2, 1.0])
    intervals = list(zip(knots[:-1], knots[1:]))
    return (
        np.concatenate([(a + b) / 2 + (b - a) * z / 2 for a, b in intervals]),
        np.concatenate([(b - a) * w / 2 for a, b in intervals]),
    )


def binary_entropy(p):
    return float(-xlogy(p, p) - xlogy(1 - p, 1 - p))


def binary_common_information(correlation):
    c = abs(float(correlation))
    if not 0 <= c <= 1:
        raise ValueError("Require a correlation in [-1, 1]")
    return (
        log(2) + binary_entropy((1 - c) / 2) - 2 * binary_entropy((1 - np.sqrt(c)) / 2)
    )


def pair_information(beta=0.8, amplitude=0.95, ramp=1.0, order=64):
    _parameters(beta, amplitude, ramp)
    u, w = quadrature(ramp, order)
    h = channel(u, amplitude, ramp)
    p = 1 + beta * h[:, None] * h[None, :]
    info = float(np.sum(np.outer(w, w) * p * np.log(p)))
    theta = abs(beta) * amplitude**2

    r = np.clip((2 * u - 1) / ramp, -1, 1)
    emission = 1 + np.sqrt(theta) * r
    one_channel_info = float(np.dot(w, emission * np.log(emission)))
    sign_correlation = theta * (1 - ramp / 2) ** 2
    lower = binary_common_information(sign_correlation)
    upper = min(binary_common_information(theta), 2 * one_channel_info - info)

    knots = np.unique([0.0, (1 - ramp) / 2, (1 + ramp) / 2, 1.0])
    score_moment = sum(
        quad(
            lambda x: ndtri(x) * np.clip((2 * x - 1) / ramp, -1, 1),
            a,
            b,
            epsabs=2e-12,
            epsrel=2e-12,
        )[0]
        for a, b in zip(knots[:-1], knots[1:])
    )
    gaussian_correlation = beta * amplitude**2 * score_moment**2
    gaussian_kl = info + 0.5 * np.log1p(-(gaussian_correlation**2))
    return dict(
        mutual_information_nats=max(0.0, info),
        channel_second_moment=amplitude**2 * (1 - 2 * ramp / 3),
        sign_correlation=sign_correlation,
        common_information_lower_nats=max(0.0, lower),
        common_information_upper_nats=max(0.0, upper),
        gaussian_correlation=gaussian_correlation,
        gaussian_KL_per_pair_nats=max(0.0, float(gaussian_kl)),
    )


def positive_width_bounds(pairs, information, epsilon):
    if (
        type(pairs) is not int
        or pairs < 0
        or not np.isfinite(information)
        or information < 0
    ):
        raise ValueError("Require a nonnegative pair count and finite information")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Require a finite nonnegative KL budget")
    kept = (
        0
        if information == 0
        else min(pairs, max(0, ceil(pairs - epsilon / information)))
    )
    return dict(
        lower_log_width_nats=max(0.0, pairs * information - epsilon),
        upper_log_width_nats=kept * log(2),
        upper_width=2**kept,
        kept_pairs=kept,
        constructive_KL_nats=(pairs - kept) * information,
    )


def sign_width_lower_bound(pairs, correlation, epsilon):
    if (
        type(pairs) is not int
        or pairs < 1
        or not np.isfinite(correlation)
        or not 0 <= correlation < 1
        or not np.isfinite(epsilon)
        or epsilon < 0
    ):
        raise ValueError("Invalid pair count, binary correlation or KL budget")
    residual = max(0.0, correlation - np.sqrt(2 * epsilon / pairs))
    return max(
        0.0,
        pairs
        * (
            log(2)
            + binary_entropy((1 - correlation) / 2)
            - 2 * binary_entropy((1 - np.sqrt(residual)) / 2)
        ),
    )


def spectral_rank(pairs, singular_value, relative_squared_error=0.01):
    if (
        type(pairs) is not int
        or not 0 <= pairs <= 64
        or not np.isfinite(singular_value)
        or not 0 <= singular_value < 1
        or not 0 <= relative_squared_error < 1
    ):
        raise ValueError("Require <=64 pairs, singular value in [0,1), error in [0,1)")
    if singular_value == 0 or pairs == 0:
        return 1
    if relative_squared_error == 0:
        return 2**pairs
    squared = singular_value**2
    total = (1 + squared) ** pairs
    remaining = (1 - relative_squared_error) * total
    rank = 0
    for k in range(pairs + 1):
        count = comb(pairs, k)
        eigenvalue = squared**k
        if remaining <= count * eigenvalue:
            return min(2**pairs, rank + max(0, ceil(remaining / eigenvalue)))
        remaining -= count * eigenvalue
        rank += count
    return 2**pairs


def paired_log_evidence(values, beta=0.8, amplitude=0.95, ramp=1.0, kept_pairs=None):
    _parameters(beta, amplitude, ramp)
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 2:
        raise ValueError("Require a nonempty batch of paired coordinates")
    m = x.shape[1] // 2
    k = m if kept_pairs is None else kept_pairs
    if type(k) is not int or not 0 <= k <= m:
        raise ValueError("Invalid retained-pair count")
    t = channel(x, amplitude, ramp)
    return np.log1p(beta * t[:, :k] * t[:, m : m + k]).sum(1)


def paired_gaussian_log_evidence(values, correlation):
    x = np.asarray(values, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] == 0
        or x.shape[1] % 2
        or not np.isfinite(correlation)
        or abs(correlation) >= 1
    ):
        raise ValueError("Require paired cube data and |correlation| < 1")
    if np.isinf(x).any() or ((x <= 0) | (x >= 1)).any():
        raise ValueError("Gaussian chart requires interior values or NaN")
    m = x.shape[1] // 2
    observed = np.isfinite(x[:, :m]) & np.isfinite(x[:, m:])
    z = ndtri(np.nan_to_num(x, nan=0.5))
    left, right = z[:, :m], z[:, m:]
    rho = correlation
    term = -0.5 * np.log1p(-rho * rho) - (
        rho * rho * (left * left + right * right) - 2 * rho * left * right
    ) / (2 * (1 - rho * rho))
    return np.where(observed, term, 0.0).sum(1)


def crossing_pairs(pairs, order, observed=None):
    if type(pairs) is not int or pairs < 1 or sorted(order) != list(range(2 * pairs)):
        raise ValueError("Require a permutation of the paired coordinates")
    seen = set()
    active = set(range(2 * pairs)) if observed is None else set(observed)
    if not active <= set(range(2 * pairs)):
        raise ValueError("Invalid observation mask")
    result = [0]
    for node in order:
        seen.add(node)
        result.append(
            sum(
                i in active
                and i + pairs in active
                and ((i in seen) != (i + pairs in seen))
                for i in range(pairs)
            )
        )
    return result

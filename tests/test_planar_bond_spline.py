import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from noisy_loop import sample_loop
from noisy_loop.planar_bond_spline import (
    PlanarBondSpline,
    channel_integral,
    emission_quantile,
)
from noisy_loop.planar_bond_transfer import transfer_log_loop
from noisy_loop.spline_memory_information import channel


def enumeration(model, x):
    spins = np.array(
        [(1,) + s for s in itertools.product([-1, 1], repeat=len(model.xy) - 1)]
    )
    bonds = spins[:, model.edges[:, 0]] * spins[:, model.edges[:, 1]]
    prior = bonds @ np.arctanh(model.prior)
    h = channel(np.asarray(x), model.amplitude, model.ramp)
    terms = prior + np.log1p(h[:, None, :] * bonds).sum(-1)
    logz = logsumexp(terms, axis=-1)
    return logz - logsumexp(prior), np.exp(terms - logz[:, None]) @ bonds


def even_subsets(model):
    result = []
    for bits in itertools.product([0, 1], repeat=model.dimension):
        parity = np.zeros(len(model.xy), int)
        for bit, (a, b) in zip(bits, model.edges):
            parity[a] ^= bit
            parity[b] ^= bit
        if not parity.any():
            result.append(np.array(bits, bool))
    return result


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("ramp", [1.0, 0.0625])
def test_masked_density_and_bond_moments_against_spin_sum(shape, ramp):
    base = PlanarBondSpline(*shape)
    rng = np.random.default_rng(57)
    model = PlanarBondSpline(
        *shape, ramp=ramp, couplings=rng.uniform(-0.7, 0.7, base.dimension)
    )
    x = rng.random((5, model.dimension))
    x[rng.random(x.shape) < 0.35] = np.nan
    expected, means = enumeration(model, x)
    assert_allclose(model.log_evidence(x), expected, atol=4e-12)
    assert_allclose(model.posterior_bonds(x), means, atol=4e-12)
    assert_allclose(
        model.log_evidence(np.full((1, model.dimension), np.nan)), 0.0, atol=1e-13
    )


def test_high_temperature_polynomial_and_derivative():
    model = PlanarBondSpline(2, 3)
    cycles = even_subsets(model)
    weights = np.random.default_rng(98).uniform(-0.94, 0.94, model.dimension)
    expected = sum(np.prod(weights[c]) for c in cycles)
    value, gradient = model.log_loop(weights, True)
    assert_allclose(np.exp(value), expected, atol=1e-13)
    for e in range(model.dimension):
        derivative = sum(
            np.prod(weights[c & (np.arange(model.dimension) != e)])
            for c in cycles
            if c[e]
        )
        assert_allclose(gradient[e], derivative / expected, atol=1e-13)


def test_forest_margins_are_independent_and_cycle_is_visible():
    model = PlanarBondSpline(3, 3)
    x = np.full((1, model.dimension), 0.9)
    x[:, 6:] = np.nan
    assert_allclose(model.log_evidence(x), 0.0, atol=1e-14)
    x[:, :] = np.nan
    square = [0, 2, 6, 7]
    x[:, square] = 0.9
    assert_allclose(model.log_evidence(x), np.log1p(model.amplitude**4), atol=1e-14)


def test_rectangles_integrate_to_one_and_match_spin_sum():
    model = PlanarBondSpline(2, 2, ramp=0.25, couplings=[0.2, -0.3, 0.1, 0.4])
    lo = np.array(list(itertools.product([0.0, 0.41], repeat=4)))
    hi = np.where(lo == 0, 0.41, 1.0)
    probabilities = np.exp(model.log_rectangle(lo, hi))
    assert_allclose(probabilities.sum(), 1.0, atol=1e-13)
    spins = np.array([(1,) + s for s in itertools.product([-1, 1], repeat=3)])
    bonds = spins[:, model.edges[:, 0]] * spins[:, model.edges[:, 1]]
    logw = bonds @ np.arctanh(model.prior)
    w = np.exp(logw - logsumexp(logw))
    integral = channel_integral(hi, model.amplitude, model.ramp) - channel_integral(
        lo, model.amplitude, model.ramp
    )
    expected = (
        np.prod((hi - lo)[:, None, :] + integral[:, None, :] * bonds, axis=-1) @ w
    )
    assert_allclose(probabilities, expected, atol=1e-13)


def test_leave_one_out_scores_match_density_ratios():
    model = PlanarBondSpline(3, 3, couplings=np.linspace(-0.5, 0.5, 12))
    x = np.random.default_rng(15).random((2, 12))
    actual = model.conditional_log_prob(x)
    for e in range(12):
        missing = x.copy()
        missing[:, e] = np.nan
        assert_allclose(
            actual[:, e],
            model.log_evidence(x) - model.log_evidence(missing),
            atol=3e-12,
        )


@pytest.mark.parametrize("ramp", [1.0, 0.25, 0.0625])
def test_scalar_inverse(ramp):
    p = np.linspace(0, 1, 1001)[:, None]
    means = np.linspace(-1, 1, 9)
    x = emission_quantile(p, means, 0.999, ramp)
    assert_allclose(
        x + means * channel_integral(x, 0.999, ramp),
        np.broadcast_to(p, x.shape),
        atol=3e-14,
    )


def test_conditional_sampler_and_observed_coordinates():
    model = PlanarBondSpline(2, 2, ramp=0.25, couplings=[0.2, -0.4, 0.3, -0.1])
    evidence = np.array([0.9, np.nan, 0.15, np.nan])
    samples = model.sample(3000, 92, evidence)
    assert_allclose(samples[:, [0, 2]], np.broadcast_to(evidence[[0, 2]], (3000, 2)))
    _, mean = enumeration(model, evidence[None])
    moment = model.amplitude**2 * (1 - 2 * model.ramp / 3)
    assert_allclose(
        channel(samples, model.amplitude, model.ramp).mean(0)[[1, 3]],
        mean[0, [1, 3]] * moment,
        atol=0.04,
    )

    first_evidence = samples.copy()
    first_evidence[:, 3] = np.nan
    _, conditional_mean = enumeration(model, first_evidence)
    pit = samples[:, 3] + conditional_mean[:, 3] * channel_integral(
        samples[:, 3], model.amplitude, model.ramp
    )
    assert abs(pit.mean() - 0.5) < 0.025
    assert abs((pit * pit).mean() - 1 / 3) < 0.025


def test_unconditional_sampler_cycle_moment():
    model = PlanarBondSpline(2, 2, ramp=0.25)
    x = model.sample(30000, 176)
    h = channel(x, model.amplitude, model.ramp)
    v = model.amplitude**2 * (1 - 2 * model.ramp / 3)
    assert abs(np.prod(h, axis=-1).mean() - v**4) < 0.012
    assert np.max(np.abs(h.mean(0))) < 0.015


def test_weighted_spectrum_and_effective_rank_bound():
    model = PlanarBondSpline(3, 3, ramp=0.25)
    cycles = even_subsets(model)
    v = model.amplitude**2 * (1 - 2 * model.ramp / 3)
    matrix = np.zeros((64, 64))
    for subset in cycles:
        i = int(subset[:6] @ (2 ** np.arange(6)))
        j = int(subset[6:] @ (2 ** np.arange(6)))
        matrix[i, j] = v ** (subset.sum() / 2)
    singular = np.linalg.svd(matrix, compute_uv=False)
    expected = sorted([v ** (c.sum() / 2) for c in cycles], reverse=True)
    assert_allclose(singular[:16], expected, atol=1e-14)
    assert_allclose(singular[16:], 0, atol=1e-14)
    record = model.spectral_certificate(0.01)
    assert record["exact_log2_rank"] == 4
    assert_allclose(
        record["log_squared_L2_norm"], np.log(np.dot(singular, singular)), atol=1e-13
    )
    assert_allclose(
        record["log_fourth_spectral_sum"], np.log(np.sum(singular**4)), atol=1e-13
    )
    rank = np.searchsorted(np.cumsum(singular**2), 0.99 * np.sum(singular**2)) + 1
    assert record["approximate_log2_rank_lower"] <= np.log2(rank) + 1e-12


def test_positive_transfer_oracle():
    model = PlanarBondSpline(4, 5)
    for amplitude in [0.1, 0.6, 0.95]:
        w = np.random.default_rng(39).uniform(-amplitude, amplitude, model.dimension)
        assert_allclose(model.log_loop(w), transfer_log_loop(model, w), atol=2e-12)


def test_cross_overlap_and_moment_fit():
    model = PlanarBondSpline(2, 2, 0.9, 0.25)
    other = PlanarBondSpline(2, 2, 0.8, 0.0625)
    inner = 0.9 * 0.8 * (1 - 0.25 / 2 - 0.0625**2 / (6 * 0.25))
    assert_allclose(model.log_overlap(other), np.log1p(inner**4), atol=1e-13)
    assert_allclose(
        model.log_overlap(model),
        model.spectral_certificate()["log_squared_L2_norm"],
        atol=1e-13,
    )
    estimated = model.fit_amplitude(model.sample(30000, 90))
    assert abs(estimated - model.amplitude) < 0.012
    independent = PlanarBondSpline(3, 3, amplitude=0.0)
    assert independent.spectral_certificate()["exact_log2_rank"] == 0
    assert_allclose(independent.log_evidence(np.full((1, 12), 0.9)), 0.0, atol=1e-14)


def test_frustrated_nearly_deterministic_input_is_rejected():
    model = PlanarBondSpline(16, 16, 0.9999)
    x = np.random.default_rng(5).random((1, model.dimension))
    with pytest.raises(FloatingPointError, match="positive-real phase"):
        model.log_evidence(x)


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "amplitude",
        "ramp",
        "couplings",
        "input",
        "interval",
        "quantile",
        "spectral",
        "sample",
    ],
)
def test_invalid_inputs(case):
    model = PlanarBondSpline(2, 2)
    with pytest.raises(ValueError):
        if case == "shape":
            PlanarBondSpline(1, 2)
        elif case == "amplitude":
            PlanarBondSpline(2, 2, amplitude=1.0)
        elif case == "ramp":
            PlanarBondSpline(2, 2, ramp=0.0)
        elif case == "couplings":
            PlanarBondSpline(2, 2, couplings=[1.0])
        elif case == "input":
            model.log_evidence([[0.0, 2.0, 0.0, 0.0]])
        elif case == "interval":
            model.log_rectangle([[0.0] * 4], [[0.0] * 4])
        elif case == "quantile":
            emission_quantile(0.5, 1.1, 0.9, 0.1)
        elif case == "spectral":
            model.spectral_certificate(1.0)
        elif case == "sample":
            model.sample(0, 1)


@pytest.mark.parametrize(
    "target,n",
    [("unknown", 4), ("ramp", 0), ("ramp", -1), ("smooth", 1.5), ("smooth", True)],
)
def test_sample_loop_rejects_invalid_requests(target, n):
    with pytest.raises(ValueError, match="target and a positive integer sample count"):
        sample_loop(4, target, 0.83125, n, 97)

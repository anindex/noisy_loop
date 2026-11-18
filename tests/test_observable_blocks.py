from fractions import Fraction
from math import prod

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from noisy_loop.observable_blocks import (
    _short_calibration,
    fit_blocks,
    grid_weights,
    log_block_evidence,
    masses,
    project_nodes,
    short_witnesses,
    witnesses,
)
from noisy_loop.planar_bond_spline import PlanarBondSpline, grid
from noisy_loop.planar_channel_study import sample, truth_for


def test_cycle_witnesses_cover_all_coordinates():
    for side in [4, 5, 8, 16, 32]:
        _, edges = grid(side, side)
        rows = witnesses(side)
        assert sorted(e for b, *_ in rows for e in b) == list(range(len(edges)))
        assert sum(len(b) == 2 for b, *_ in rows) == 4
        eta = [Fraction(2 + e % 7, 11) for e in range(len(edges))]
        for b, c, d, q in rows:
            theta = lambda cycle: prod(eta[e] for e in cycle)
            assert theta(c) * theta(q) / theta(d) == theta(set(c) - set(b)) ** 2
            assert set(c) & set(d) == set(b)
            for cycle in [c, d, q]:
                assert (
                    np.bincount(edges[cycle].ravel(), minlength=side**2) % 2 == 0
                ).all()
            assert max(map(len, [c, d, q])) <= 12


def test_short_calibration_geometry_and_population_scales():
    for side in [4, 5, 8, 16]:
        _, edges = grid(side, side)
        eta = np.random.default_rng(side).uniform(0.25, 0.85, len(edges))
        rows = short_witnesses(side)
        assert [r[0] for r in rows] == [r[0] for r in witnesses(side)]
        for block, face, interior, cycles in rows:
            assert set(block) <= set(face)
            assert max(map(len, cycles)) == 6
            assert len(set().union(*map(set, cycles))) <= 13
            for cycle in cycles:
                assert (
                    np.bincount(edges[list(cycle)].ravel(), minlength=side**2) % 2 == 0
                ).all()
            m = [np.prod(eta[list(c)]) for c in cycles]
            if interior:
                assert set(interior) == set(face) - set(block)
                recovered = [
                    np.sqrt(m[0] * m[2 * j + 1] / m[2 * j + 2])
                    for j in range(len(interior))
                ]
                np.testing.assert_allclose(recovered, eta[list(interior)], atol=1e-14)
                scale = np.prod(recovered)
            else:
                scale = np.sqrt(m[0] * m[2] / m[1])
            np.testing.assert_allclose(
                scale, np.prod(eta[list(set(face) - set(block))]), atol=1e-14
            )


@pytest.mark.parametrize("sample_split", [True, False])
def test_short_fit_clipping_and_normalized_corner_blocks(sample_split):
    side = 4
    rng = np.random.default_rng(392)
    x = rng.uniform(size=(512, 24))
    rows = _short_calibration(np.where(x[:256] > 0.5, 1.0, -1.0), side, 0.2, 0.8)
    for block, face, scale, theta, diagnostic in rows:
        assert 0.2 ** (4 - len(block)) <= scale <= 1
        assert theta >= 0.2**4
        assert max(diagnostic["lengths"]) <= 6
    fitted, diagnostics = fit_blocks(
        x, side, 4, kappa=0.2, bound=0.8, calibration="short", sample_split=sample_split
    )
    for block, v in fitted.items():
        w = masses(4)
        np.testing.assert_allclose(v @ w, 0, atol=1e-12)
        if len(block) == 2:
            np.testing.assert_allclose(w @ v, 0, atol=1e-12)
        assert np.max(np.abs(v)) <= 0.8 ** len(block) + 1e-12
    assert all(d["calibration"] == "short" for d in diagnostics)
    assert sum(d["clipped"] for d in diagnostics) > 0
    with pytest.raises(ValueError):
        fit_blocks(x, side, 4, calibration="unknown")


def test_default_fit_reuses_all_vectors_and_legacy_fit_is_preserved():
    from pathlib import Path

    x = PlanarBondSpline(4, 4, amplitude=0.8).sample(1024, 4606)
    options = dict(side=4, cells=4, kappa=0.2)
    full, _ = fit_blocks(x, **options, calibration="short", sample_split=False)
    duplicated, _ = fit_blocks(np.concatenate([x, x]), **options, sample_split=True)
    permuted, _ = fit_blocks(x[::-1], **options, sample_split=False)
    default, _ = fit_blocks(x, **options)
    legacy, _ = fit_blocks(x, **options, calibration="legacy", sample_split=True)
    saved = np.load(Path(__file__).parent / "fixtures/legacy-fit.npz")
    for block in full:
        np.testing.assert_allclose(full[block], duplicated[block], atol=1e-12)
        np.testing.assert_allclose(full[block], permuted[block], atol=1e-10)
        np.testing.assert_array_equal(default[block], full[block])
        np.testing.assert_array_equal(legacy[block], saved[",".join(map(str, block))])


def test_projection_and_corner_lift():
    rng = np.random.default_rng(123)
    for shape in [(9,), (9, 9)]:
        raw = rng.normal(size=shape) * 4
        v = project_nodes(raw, 0.8)
        w = masses(8)
        assert np.max(np.abs(v)) <= 0.8 + 1e-10
        np.testing.assert_allclose(v @ w, 0, atol=1e-10)
        if v.ndim == 2:
            np.testing.assert_allclose(w @ v, 0, atol=1e-10)
        np.testing.assert_allclose(project_nodes(v, 0.8), v, atol=1e-10)
    graph = PlanarBondSpline(4, 4)
    a, ramp, _ = truth_for(graph, 11)
    x, _ = sample(graph, a, ramp, 1024, 9)
    fitted, diagnostics = fit_blocks(x, 4, 4)
    assert len(fitted) == 20
    weights = grid_weights(fitted, x[:3])
    alternative = weights.copy()
    for b, *_ in witnesses(4):
        if len(b) == 2:
            product = weights[:, b[0]] * weights[:, b[1]]
            alternative[:, b[0]] = 0.99
            alternative[:, b[1]] = product / 0.99
    np.testing.assert_allclose(
        [graph.log_loop(w) for w in weights],
        [graph.log_loop(w) for w in alternative],
        atol=1e-12,
    )
    assert all(r["projection"]["gap"] <= 1e-9 for r in diagnostics)
    assert all(
        r["squared_estimator_error_bound"] >= 2 * r["projection"]["gap"] - 1e-20
        for r in diagnostics
    )


def test_general_corner_tensors_preserve_forest_marginals():
    graph = PlanarBondSpline(4, 4)
    rng = np.random.default_rng(4781)
    fitted = {
        b: project_nodes(rng.normal(size=(5,) * len(b)), 0.7 ** len(b))
        for b, *_ in witnesses(4)
    }
    points = rng.random((24, graph.dimension))
    for block, face, *_ in witnesses(4):
        if len(block) != 2:
            continue
        assert np.linalg.matrix_rank(fitted[block], tol=1e-10) > 1
        third = next(e for e in face if e not in block)
        for forest in [list(block), [*block, third]]:
            lo, hi = np.zeros_like(points), np.ones_like(points)
            lo[:, forest] = hi[:, forest] = points[:, forest]
            np.testing.assert_allclose(
                log_block_evidence(graph, fitted, lo, hi), 0, atol=1e-12
            )
        lo, hi = np.zeros_like(points), np.ones_like(points)
        lo[:, face] = hi[:, face] = points[:, face]
        assert np.max(np.abs(log_block_evidence(graph, fitted, lo, hi))) > 1e-5


def test_conditional_queries_and_kl_chain_rule_by_exact_bin_enumeration():
    from itertools import product

    graph = PlanarBondSpline(4, 4)
    corner, face, *_ = next(row for row in witnesses(4) if len(row[0]) == 2)
    active = [*corner, *sorted(set(face) - set(corner))]
    ramp = np.linspace(-0.8, 0.8, 4)
    truth = {b: np.zeros((4,) * len(b)) for b, *_ in witnesses(4)}
    truth[corner] = np.outer(ramp, ramp)
    truth[active[2],] = ramp
    truth[active[3],] = ramp * 0.9
    fitted = {b: v.copy() for b, v in truth.items()}
    fitted[corner] = project_nodes(np.random.default_rng(91).normal(size=(4, 4)), 0.7)
    assert np.linalg.matrix_rank(fitted[corner], tol=1e-10) > 1
    bins = np.array(list(product(range(3), repeat=4)))
    lower, upper = np.zeros((len(bins), 24)), np.ones((len(bins), 24))
    lower[:, active], upper[:, active] = bins / 3, (bins + 1) / 3

    def probabilities(nodes):
        c = nodes[corner]
        c = (c[:-1, :-1] + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:]) / 4
        s, t = [nodes[e,] for e in active[2:]]
        s, t = (s[:-1] + s[1:]) / 2, (t[:-1] + t[1:]) / 2
        return (1 + c[bins[:, 0], bins[:, 1]] * s[bins[:, 2]] * t[bins[:, 3]]) / 81

    p, q = probabilities(truth), probabilities(fitted)
    lp = log_block_evidence(graph, truth, lower, upper)
    lq = log_block_evidence(graph, fitted, lower, upper)
    np.testing.assert_allclose(np.exp(lp), p, atol=1e-14)
    np.testing.assert_allclose(np.exp(lq), q, atol=1e-14)
    joint = np.sum(p * np.log(p / q))
    horizontal = [e for e in active if e < graph.dimension // 2]
    for observed in [[], list(range(24)), active, horizontal, [corner[0], active[2]]]:
        lo, hi = np.zeros_like(lower), np.ones_like(upper)
        lo[:, observed], hi[:, observed] = lower[:, observed], upper[:, observed]
        columns = [i for i, edge in enumerate(active) if edge in observed]
        same = (bins[:, None, columns] == bins[None, :, columns]).all(axis=2)
        po, qo = same @ p, same @ q
        lpo = log_block_evidence(graph, truth, lo, hi)
        lqo = log_block_evidence(graph, fitted, lo, hi)
        np.testing.assert_allclose(np.exp(lpo), po, atol=1e-13)
        np.testing.assert_allclose(np.exp(lqo), qo, atol=1e-13)
        conditional = np.sum(p * ((lp - lpo) - (lq - lqo)))
        marginal = np.sum(p * np.log(po / qo))
        assert conditional == pytest.approx(joint - marginal, abs=1e-13)
        assert -1e-13 <= conditional <= joint + 1e-13
        np.testing.assert_allclose(same @ np.exp(lq - lqo), 1, atol=1e-13)


def test_projection_gap_against_independent_quadratic_program():
    rng = np.random.default_rng(8124)
    for shape in [(5,), (5, 5)]:
        v = rng.normal(size=shape) * 4
        w = masses(4)
        weight = w if v.ndim == 1 else np.outer(w, w)
        if v.ndim == 1:
            equality = w[None, :]
        else:
            equality = np.r_[
                np.kron(np.eye(5), w[None, :]), np.kron(w[None, :], np.eye(5))[:-1]
            ]
        result = minimize(
            lambda z: 0.5 * np.sum(weight.ravel() * (z - v.ravel()) ** 2),
            np.zeros(v.size),
            jac=lambda z: weight.ravel() * (z - v.ravel()),
            constraints=[LinearConstraint(equality, 0, 0)],
            bounds=[(-0.8, 0.8)] * v.size,
            method="SLSQP",
            options=dict(ftol=1e-13, maxiter=2000),
        )
        assert result.success, result.message
        for tolerance in [1e-2, 1e-9]:
            x, info = project_nodes(
                v, 0.8, return_diagnostics=True, gap_tolerance=tolerance
            )
            assert info["gap"] <= tolerance
            assert info["centering_residual"] < 1e-12
            assert info["amplitude_violation"] == 0
            assert (
                0.5 * np.sum(weight * (x - v) ** 2) - result.fun <= info["gap"] + 1e-11
            )
            assert (
                np.sum(weight * (x - result.x.reshape(shape)) ** 2)
                <= 2 * info["gap"] + 1e-11
            )
    with pytest.raises(FloatingPointError, match="tolerance unmet"):
        project_nodes(v, 0.8, max_iterations=1, gap_tolerance=1e-12)
    for options in [
        dict(gap_tolerance=0),
        dict(gap_tolerance=np.nan),
        dict(max_iterations=0),
    ]:
        with pytest.raises(ValueError):
            project_nodes(v, 0.8, **options)

"""CPU experiments for fitting, conditional prediction, tiling and query cost."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

from noisy_loop import PlanarBondSpline, fit_blocks, log_block_evidence, sample_loop
from noisy_loop.noisy_loop_calibration import block_error, curve, integration
from noisy_loop.planar_bond_transfer import transfer_log_loop


def summary(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite experiment score")
    return dict(
        mean=float(values.mean()),
        mc_se=float(values.std(ddof=1) / np.sqrt(len(values))),
    )


def masks(graph):
    midpoint = graph.xy[graph.edges].mean(1)
    order = np.random.default_rng(
        np.random.SeedSequence([20261104, 591, graph.rows])
    ).permutation(graph.dimension)
    result = {}
    for percent in [25, 50]:
        observed = np.ones(graph.dimension, bool)
        observed[order[: graph.dimension * percent // 100]] = False
        result[f"random{percent}"] = observed
    result["patch"] = ~((midpoint < graph.rows / 2).all(1))
    result["forest"] = np.arange(graph.dimension) < graph.dimension // 2
    return result


def fitted_scores(graph, nodes, x, observed):
    lower, upper = x.copy(), x.copy()
    lower[:, ~observed], upper[:, ~observed] = 0, 1
    return log_block_evidence(graph, nodes, lower, upper)


def fit(args):
    for side in args.sides:
        graph = PlanarBondSpline(side, side)
        full = np.ones(graph.dimension, bool)
        observed_masks = masks(graph)
        for target in args.targets:
            for eta in args.etas:
                for seed in args.seeds:
                    streams = np.random.SeedSequence([seed, side]).spawn(3)
                    train = sample_loop(side, target, eta, max(args.counts), streams[0])
                    test = sample_loop(side, target, eta, args.test, streams[2])
                    if args.shrink:
                        validation = sample_loop(
                            side, target, eta, args.validation, streams[1]
                        )
                    weights = curve(test, target, eta)
                    truth = np.array([graph.log_loop(row) for row in weights])
                    evidence = {
                        name: np.array(
                            [graph.log_loop(row) for row in weights * observed]
                        )
                        for name, observed in observed_masks.items()
                    }
                    for count in args.counts:
                        growing = 8 * int(np.ceil(np.cbrt(count) / 8))
                        resolutions = dict(fixed=args.cells, growing=growing)
                        if args.resolution != "both":
                            resolutions = {
                                args.resolution: resolutions[args.resolution]
                            }
                        for resolution, cells in resolutions.items():
                            integrals = integration(target, eta, cells)
                            started = time.perf_counter()
                            nodes, diagnostics = fit_blocks(
                                train[:count],
                                side,
                                cells,
                                kappa=args.kappa,
                                bound=args.bound,
                            )
                            seconds = time.perf_counter() - started
                            raw = fitted_scores(graph, nodes, test, full)
                            conditional = {
                                name: summary(
                                    truth
                                    - truth_o
                                    - raw
                                    + fitted_scores(
                                        graph, nodes, test, observed_masks[name]
                                    )
                                )
                                for name, truth_o in evidence.items()
                            }
                            row = dict(
                                side=side,
                                dimension=graph.dimension,
                                target=target,
                                eta=eta,
                                seed=seed,
                                count=count,
                                cells=cells,
                                resolution=resolution,
                                joint_kl=summary(truth - raw),
                                independence_kl=summary(truth),
                                conditional_kl=conditional,
                                fit_seconds=seconds,
                                block_ise=sum(
                                    block_error(v, *integrals) for v in nodes.values()
                                ),
                                projection_gap=max(
                                    d["projection"]["gap"] for d in diagnostics
                                ),
                            )
                            if args.shrink:
                                strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
                                scaled = lambda strength: {
                                    b: strength ** len(b) * v for b, v in nodes.items()
                                }
                                scores = [
                                    fitted_scores(
                                        graph, scaled(s), validation, full
                                    ).mean()
                                    for s in strengths
                                ]
                                strength = strengths[int(np.argmax(scores))]
                                row.update(
                                    shrinkage=strength,
                                    shrunk_kl=summary(
                                        truth
                                        - fitted_scores(
                                            graph, scaled(strength), test, full
                                        )
                                    ),
                                )
                            yield row


def tiling(args):
    for side in args.sides:
        graph = PlanarBondSpline(side, side, amplitude=0.95, ramp=0.25)
        start, width = (side - side // 2) // 2, side // 2
        inside = ((graph.xy >= start) & (graph.xy < start + width)).all(1)
        hidden = inside[graph.edges].all(1)
        observed = ~hidden
        for seed in args.seeds:
            x = graph.sample(args.test, seed)
            weights = curve(x, "ramp", 0.83125)
            truth = np.array([graph.log_loop(row) for row in weights])
            truth_o = np.array([graph.log_loop(row) for row in weights * observed])
            widths = args.widths or sorted({2, 4, int(np.log2(graph.dimension))})
            for width in widths:
                tiles = graph.xy // width
                deleted = (tiles[graph.edges[:, 0]] != tiles[graph.edges[:, 1]]).any(1)
                tiled = weights.copy()
                tiled[:, deleted] = 0
                score = np.array([graph.log_loop(row) for row in tiled])
                score_o = np.array([graph.log_loop(row) for row in tiled * observed])
                loss = truth - score
                yield dict(
                    side=side,
                    dimension=graph.dimension,
                    seed=seed,
                    tile_width=width,
                    deleted_edges=int(deleted.sum()),
                    hidden_edges=int(hidden.sum()),
                    joint_kl=summary(loss),
                    tv=summary(-np.expm1(np.minimum(-loss, 0))),
                    conditional_kl=summary(truth - truth_o - score + score_o),
                    kl_upper=0.95**2 * (1 - 2 * 0.25 / 3) * int(deleted.sum()),
                )


def separator_order(graph):
    def groups(r0, r1, c0, c1):
        if r0 == r1 or c0 == c1:
            return []
        if r1 - r0 >= c1 - c0:
            mid = (r0 + r1) // 2
            return (
                groups(r0, mid, c0, c1)
                + groups(mid + 1, r1, c0, c1)
                + [[(mid, c) for c in range(c0, c1)]]
            )
        mid = (c0 + c1) // 2
        return (
            groups(r0, r1, c0, mid)
            + groups(r0, r1, mid + 1, c1)
            + [[(r, mid) for r in range(r0, r1)]]
        )

    labels = np.empty((graph.rows, graph.cols), int)
    for rank, group in enumerate(groups(0, graph.rows, 0, graph.cols)):
        for row, col in group:
            labels[row, col] = rank
    rank = labels.ravel()[graph.edges].max(1)
    return np.argsort(np.tile(rank, 2), kind="stable")


def queries(args):
    for side in args.sides:
        graph = PlanarBondSpline(side, side)
        for seed in args.seeds:
            x = graph.sample(1, seed)
            weights = graph.amplitude * np.clip((2 * x[0] - 1) / graph.ramp, -1, 1)
            matrix = graph.identity - graph.turn.multiply(np.tile(weights, 2)).tocsc()
            matrix.eliminate_zeros()
            reference = graph.log_loop(weights)
            transfer = transfer_log_loop(graph, weights) if side <= 8 else None
            order = separator_order(graph)
            for name in ["colamd", "nested_dissection"]:
                started = time.perf_counter()
                factor = (
                    splu(matrix)
                    if name == "colamd"
                    else splu(
                        matrix[order][:, order].tocsc(),
                        permc_spec="NATURAL",
                        diag_pivot_thresh=0,
                        options=dict(SymmetricMode=True),
                    )
                )
                seconds = time.perf_counter() - started
                value = float(0.5 * np.log(np.abs(factor.U.diagonal())).sum())
                if not np.isfinite(value) or abs(value - reference) > 1e-7 * max(
                    1, abs(reference)
                ):
                    raise FloatingPointError("Factorization orders disagree")
                yield dict(
                    side=side,
                    dimension=graph.dimension,
                    seed=seed,
                    ordering=name,
                    seconds=seconds,
                    fill=int(factor.L.nnz + factor.U.nnz),
                    multiplies=int(
                        np.diff(factor.L.tocsc().indptr).astype(np.int64)
                        @ np.diff(factor.U.tocsr().indptr).astype(np.int64)
                    ),
                    log_likelihood=value,
                    order_discrepancy=abs(value - reference),
                    transfer_discrepancy=None
                    if transfer is None
                    else abs(value - transfer),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["fit", "tiling", "queries"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sides", type=int, nargs="+", default=[4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--counts", type=int, nargs="+", default=[8192])
    parser.add_argument(
        "--targets", choices=["ramp", "smooth"], nargs="+", default=["ramp"]
    )
    parser.add_argument("--etas", type=float, nargs="+", default=[0.83125])
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument(
        "--resolution", choices=["fixed", "growing", "both"], default="fixed"
    )
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--bound", type=float, default=0.98)
    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--validation", type=int, default=4096)
    parser.add_argument("--widths", type=int, nargs="+")
    args = parser.parse_args()
    if (
        min(args.sides) < 4
        or min(args.seeds) < 0
        or min(args.counts) < 4
        or args.test < 2
        or args.validation < 2
        or args.cells < 1
        or args.widths is not None
        and min(args.widths) < 1
    ):
        parser.error(
            "Require sides >= 4, nonnegative seeds, counts >= 4, test/validation >= 2 and positive widths/cells"
        )
    if not 0 < args.kappa < args.bound < 1 or not all(
        args.kappa <= eta and eta / 0.875 <= args.bound for eta in args.etas
    ):
        parser.error("Require 0 < kappa <= eta <= 0.875*bound and kappa < bound < 1")
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    rows = list({"fit": fit, "tiling": tiling, "queries": queries}[args.study](args))
    options = vars(args).copy()
    options["out"] = str(args.out)
    payload = (
        json.dumps(dict(settings=options, rows=rows), indent=2, allow_nan=False) + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(payload)
    print(f"{len(rows)} results written to {args.out}")


if __name__ == "__main__":
    main()

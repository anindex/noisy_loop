# Continuous noisy-loop models

We learn continuous noise channels on a known grid and answer density, interval and missing-reading queries.

Four hidden spins sit at the corners of a square. Each edge reports whether its two spins agree. The product of the four true edge signs is always positive, so three noisy readings can help predict the fourth. Every pair of readings is independent in the model. Learning this dependence requires information from the whole cycle.

<img src="docs/assets/noisy-loop-hero.gif" width="1280" alt="Four saved fits on a 24-edge grid: learned channels approach the true channel as training size grows, while pairwise sign probabilities stay uniform.">

The animation shows four fits from one seed, using 8,192 to 524,288 training vectors. Teal curves represent the highlighted edges. Positive channel values favor spin agreement; negative values favor disagreement. Joint KL measures distribution error; lower is better. We estimate it on 65,536 held-out vectors, with ±1.96 Monte Carlo standard errors for each fixed fit. “Independent” is the uniform product baseline.

In the final panel, the white outline tracks the observed signs. The fitted model assigns about 74% probability to an even number of negative signs, compared with 50% under independence.

See the [experiment guide](docs/reproduction.md) for study commands, score definitions and data sources.

## Install

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). NumPy and SciPy are the runtime dependencies; examples run on CPU.

```bash
git clone https://github.com/anindex/noisy_loop.git
cd noisy_loop
uv sync --locked
```

The lockfile pins the tested dependencies. Run commands with `uv run`; no environment activation is needed. To install the package in another project's environment:

```bash
uv add "noisy-loop @ git+https://github.com/anindex/noisy_loop.git"
```

## Examples

Run from the repository root:

```bash
uv run python -m examples.noisy_loop_exact
OMP_NUM_THREADS=1 uv run python -m examples.noisy_loop_fit
uv run python -m examples.noisy_loop_readout
```

| Example | What it shows |
|---|---|
| [Exact queries](examples/noisy_loop_exact.py) | A determinant query agrees with enumeration over 512 spin states on a 3×3 grid. Forest marginals are independent. |
| [Learning channels](examples/noisy_loop_fit.py) | Fit 24 readings per vector from 8,192 synthetic training vectors. Cycle moments calibrate the channels; spline regression estimates their shapes. Check missing-reading and interval queries against positive row transfer. |
| [Measured readout](examples/noisy_loop_readout.py) | Fit noise channels from measured superconducting-qubit IQ pairs, then predict bonds on a synthetic square using held-out measurements. |

The readout example uses measured calibration data from [Hanisch, Hetényi and Wootton](https://arxiv.org/abs/2411.16228) on a synthetic square. Each IQ pair contains the in-phase and quadrature components of one readout. The first reading gives a local likelihood ratio; the other three add information through the square's parity constraint. Log loss is 0.045615 nats per bond using the first reading alone and 0.015850 using all four. We check the posterior against spin enumeration and the determinant gradient.

This example uses supervised Gaussian calibration and resampled measurements; it is not a hardware loop experiment. The spline example learns channels from unlabelled observation vectors. The guide gives the [calibration protocol and data license](docs/reproduction.md#example-data).

## Fit and query

```python
from noisy_loop import PlanarBondSpline, fit_blocks, log_block_evidence, sample_loop

train = sample_loop(side=4, target='ramp', eta=0.83125, n=8192, seed=1)
test = sample_loop(side=4, target='ramp', eta=0.83125, n=1024, seed=2)
nodes, diagnostics = fit_blocks(train, side=4, cells=8, kappa=0.2, bound=0.98)
graph = PlanarBondSpline(4, 4)

log_density = log_block_evidence(graph, nodes, test, test)
lower, upper = test.copy(), test.copy()
lower[:, ::2], upper[:, ::2] = 0.0, 1.0
log_observed = log_block_evidence(graph, nodes, lower, upper)
log_conditional = log_density - log_observed
```

Rows are independent observation vectors; columns follow `graph.edges`, horizontal edges first. Equal bounds specify a point density, `[0, 1]` integrates out a coordinate, and other bounds specify an interval. The last line scores the even-indexed readings given the remaining readings, in nats per vector.

Use `target='ramp'` or `target='smooth'` and a positive integer sample count `n`.

| Argument | Paper notation | Meaning |
|---|---|---|
| `side` | $L$ | Grid side in vertices; $d=2L(L-1)$ |
| `cells` | $K$ | Linear-spline cells per channel |
| `bound` | $a_{\mathrm{in}}$ | Input amplitude cap, with $a\leq a_{\mathrm{in}}<1$ |
| `kappa` | $\kappa_{\mathrm{in}}$ | Positive input anchor, no greater than the true reliability lower bound |

`fit_blocks` requires `side >= 4`, at least four finite vectors in `[0, 1]`, and `0 < kappa < bound < 1`. By default, it uses all vectors for short-cycle calibration and regression. It estimates the identifiable scalar channels and two-coordinate corner products. Fits are centered and bounded; diagnostics report feasibility residuals and a primal-dual projection gap.

## Scope and numerical limits

Fitting assumes a known square grid, known uniform margins and independent uniform latent spins. Arbitrary real data need not satisfy these assumptions. In particular, a loop law cannot represent dependence on a forest. The finite-net estimator for the paper's anchor-free rate is not implemented.

`PlanarBondSpline` also supports known ramp channels on rectangular grids, including a fixed zero-field Ising prior. This query interface does not learn that prior. `log_block_evidence` requires the independent-spin model.

The code uses floating-point sparse LU. Evidence favoring nearly incompatible bonds can make the determinant unreliable; phase and posterior-range checks detect some failures but do not certify forward error. The paper's exact-arithmetic cost excludes bit complexity. Use the positive row-transfer reference in `noisy_loop.planar_bond_transfer` on small grids when checking difficult queries. It is exponential in the grid width and accepts at most 24 columns.

## Checks

```bash
OMP_NUM_THREADS=1 uv run pytest -q
```

The tests compare likelihoods, gradients and conditional scores with spin enumeration and positive transfer, and projection gaps with an independent quadratic program. They also cover calibration, normalization, sampling and the experiment harness.

## License

Original code is Apache-2.0 ([LICENSE](LICENSE)); the measured IQ data retain their MIT license ([NOTICE](NOTICE)).

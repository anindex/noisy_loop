# Experiments

Run from the repository root after `uv sync --locked`. The CPU harness samples, fits and scores models, then writes settings and results to a new JSON file. It refuses to overwrite existing files.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit --out results/fit.json
OMP_NUM_THREADS=1 uv run python experiments.py tiling --sides 8 --out results/tiling.json
OMP_NUM_THREADS=1 uv run python experiments.py queries --sides 4 8 16 --out results/queries.json
```

Use `--help` for options. Each row identifies the grid, seed and study settings. KL scores are in nats per observation vector; divide by `dimension` for per-coordinate values.

## Fitting and conditional prediction

`fit` estimates scalar channels and corner products. It reports joint KL, conditional KL for four masks, summed block integrated squared error (ISE), projection gap and fitting time. The random masks hide 25% and 50% of coordinates using one nested permutation. The patch hides edges whose midpoints lie below half the grid side in both coordinates. The forest observes horizontal edges. These masks follow the paper's empirical comparison; they differ from the masks in the fixed-mask theorem.

Training sizes share nested prefixes; resolutions and masks share test vectors. Training, validation and test streams are independent. Add `--shrink` to select a channel scale from `{0, 0.25, 0.5, 0.75, 1}` on validation likelihood. A block with two coordinates is scaled by the square of this value.

Conditional scores and block ISE describe the fit before shrinkage. With `--shrink`, `shrunk_kl` reports joint KL for the selected scale.

For the resolution and sample-size study:

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit \
  --sides 4 8 --targets ramp smooth --kappa 0.3 \
  --counts 8192 32768 131072 524288 --resolution both \
  --seeds 1 2 3 4 5 6 7 8 9 10 --test 65536 --out results/rates.json
```

The fixed resolution uses eight cells. The growing resolution uses `8*ceil(N**(1/3)/8)` cells. Both contain the ramp breakpoints. For weak signals:

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit \
  --sides 8 --targets ramp smooth --etas 0.28 0.375 0.5 0.83125 \
  --counts 8192 32768 --kappa 0.2 --shrink --validation 4096 \
  --seeds 1 2 3 4 5 --test 65536 --out results/weak.json
```

`independence_kl` uses the same test vectors as the fitted model. The paper's weak-signal plot uses a separate, larger sample for this reference.

## Tiling and query cost

`tiling` uses known ramp channels with amplitude 0.95 and width 0.25. It deletes edges between square tiles and reports joint KL, TV, missing-patch conditional KL and the energy upper bound. The hidden patch is the induced subgraph of the centered half-side vertex square.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py tiling \
  --sides 8 12 16 24 32 --test 10000 --out results/tiling-large.json
```

By default, tile widths are 2, 4 and `floor(log2(d))`; use `--widths` to choose others. Scores share samples across widths. TV is a Monte Carlo estimate of `(1-q/p)_+`; a value near one cannot exclude rare overlap.

`queries` compares column approximate minimum degree (COLAMD) ordering and nested dissection for sparse LU with amplitude 0.95 and ramp width 1/16. It reports factorization time, fill and multiplication counts, and checks log-density agreement. Positive row transfer supplies an independent numerical reference up to side eight. Ordering time is excluded. Runtime depends on the machine and SciPy version; operation counts do not establish floating-point accuracy on larger grids.

## Reading the results

`mean` and `mc_se` describe Monte Carlo variation over independent test vectors for one fit or known model. They do not include training variation. For a comparison across seeds, use paired differences of seed means and a Student interval over those differences. Retain negative KL estimates: finite test samples can give them even though population KL is nonnegative.

The harness covers moment fitting, shrinkage, conditional scoring, tiling and query cost with new random streams. It does not regenerate every paper table or figure: the likelihood, EM, HCLT, GMM, MPC and OSOS comparison, observable-geometry study and heterogeneous-channel study are outside its scope. The paper gives those methods, budgets and results. Full-size runs can take substantial CPU time; start with the small commands above.

## Example data

[`examples/data/readout_iq.npz`](../examples/data/readout_iq.npz) contains only `cal_q3_first0`, `cal_q3_first1` and their scale arrays from the upstream `iq_figures.npz`. Each class has 15,748 measured IQ pairs. Array values and row order are unchanged; multiply the integer arrays by their respective scale to recover the measurements.

The upstream source is [quantumjim/soft_information_decoding](https://github.com/quantumjim/soft_information_decoding/tree/5ef9a62f638b5ee6b276c16f63189b1ce10540a8), commit `5ef9a62f638b5ee6b276c16f63189b1ce10540a8`, associated with [Hanisch, Hetényi and Wootton, *Soft information decoding with superconducting qubits*](https://arxiv.org/abs/2411.16228). The [MIT license](../examples/data/LICENSE) applies, copyright 2023 Maurice D. Hanisch.

We fit one Gaussian per prepared state using the first 8,000 pairs per class, then mix 2% of the opposite class into each density to limit overconfident predictions. We resample the remaining 7,748 pairs per class for evaluation. The graph and bond labels are synthetic; repeated pool draws do not create independent hardware trials. The scores describe this calibration model on a synthetic loop, rather than performance on a hardware loop experiment.

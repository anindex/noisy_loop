# Experiments

To run a small version of each study, start in the repository root after `uv sync --locked`. Each command runs on CPU and writes the settings and results to a new JSON file. Choose a fresh output path when you rerun a study.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit --out results/fit.json
OMP_NUM_THREADS=1 uv run python experiments.py tiling --sides 8 --out results/tiling.json
OMP_NUM_THREADS=1 uv run python experiments.py queries --sides 4 8 16 --out results/queries.json
```

In the JSON output, `settings` records the command options and `rows` contains the results by grid, seed and study setting. KL scores are in nats per observation vector. Divide by `dimension` for nats per coordinate. Use `--help` to see the available options.

## Fitting and conditional prediction

`fit` estimates scalar channels and corner products. It reports joint KL, conditional KL for four masks, the sum of the block integrated squared errors (ISE), the largest projection gap and fitting time.

The masks here match the paper’s experiments. The fixed-mask theorem uses a different set. Two random masks hide 25% and 50% of the coordinates, with the smaller set contained in the larger one. The patch mask hides edges whose midpoints fall below half the grid side in both coordinates. The forest mask leaves only horizontal edges visible.

For a given grid and seed, smaller training sets use the first vectors of the largest set. All resolutions and masks use the same test vectors. Training, validation and test draws use independent random streams.

With `--shrink`, the program chooses the channel scale in `{0, 0.25, 0.5, 0.75, 1}` with the highest validation likelihood. Scaling two channels by the same factor multiplies their corner product by that factor squared. The `shrunk_kl` field gives joint KL after shrinkage. The conditional scores and block ISE describe the original fit.

To see how the choice of spline resolution affects the fit as more data become available, run the sample-size study.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit \
  --sides 4 8 --targets ramp smooth --kappa 0.3 \
  --counts 8192 32768 131072 524288 --resolution both \
  --seeds 1 2 3 4 5 6 7 8 9 10 --test 65536 --out results/rates.json
```

At each of the four training sizes, the study compares a fixed grid of eight cells with a grid of `8*ceil(N**(1/3)/8)` cells for `N` training vectors. Both grids contain the ramp breakpoints. The weak-signal study below varies reliability and uses validation data to choose the shrinkage.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py fit \
  --sides 8 --targets ramp smooth --etas 0.28 0.375 0.5 0.83125 \
  --counts 8192 32768 --kappa 0.2 --shrink --validation 4096 \
  --seeds 1 2 3 4 5 --test 65536 --out results/weak.json
```

Here `independence_kl` uses the same test vectors as the fitted model. For the independence reference in the paper’s weak-signal plot, I used a separate, larger sample.

## Tiling and query cost

`tiling` starts with known ramp channels of amplitude 0.95 and width 0.25, then deletes the edges between square tiles. It reports joint KL, total variation (TV), conditional KL for a missing patch and the energy upper bound. The patch hides all edges with both endpoints in a centered square of vertices half as wide as the grid.

```bash
OMP_NUM_THREADS=1 uv run python experiments.py tiling \
  --sides 8 12 16 24 32 --test 10000 --out results/tiling-large.json
```

The default tile widths are 2, 4 and `floor(log2(d))`, where `d` is the number of edges. Use `--widths` to change them. All widths use the same samples. For TV, the program averages `max(0, 1 - q/p)` over draws from the full model `p`, with `q` the tiled model. An estimate near one can miss rare samples where the two models overlap.

`queries` compares column approximate minimum degree (COLAMD) ordering with nested dissection for sparse LU, using amplitude 0.95 and ramp width 1/16. It checks that the log densities agree and reports elapsed time, fill and multiplication counts. SciPy chooses the COLAMD ordering inside the timed call. For nested dissection, the script computes the order before timing the matrix permutation and factorization. Timings depend on the machine and SciPy version. Positive row transfer gives an independent accuracy check through side eight. On larger grids the script checks agreement between orderings.

## Reading the results

`mean` is the average over independent test vectors for one fit or known model. Its Monte Carlo standard error, `mc_se`, measures test-sample variation with that model held fixed. To compare fits across seeds, take the difference between the two means for each seed, then compute a Student’s t interval across those differences. Keep negative KL estimates in the comparison. Sampling variation can put an estimate below zero even when the population KL is positive.

The fitting, shrinkage, conditional prediction, tiling and query-cost commands use different random streams from the paper, so individual estimates will vary. Full-size runs can take substantial CPU time. The numerical appendices cover the studies outside this script’s scope, including the likelihood, EM, HCLT, GMM, MPC and OSOS comparison, the observable-geometry study and the heterogeneous-channel study. [Methods, budgets and results](https://arxiv.org/pdf/2611.24213)

## Example data

The bundled archive contains `cal_q3_first0`, `cal_q3_first1` and their scale arrays from the upstream `iq_figures.npz`. Each prepared state has 15,748 measured IQ pairs in their original order. Multiply each integer array by its scale to recover the measurements. [Calibration data](../examples/data/readout_iq.npz)

The measurements come from Hanisch, Hetényi and Wootton’s *Soft information decoding with superconducting qubits*, at commit `5ef9a62f638b5ee6b276c16f63189b1ce10540a8` of their data repository. The data are copyright 2023 Maurice D. Hanisch and distributed under the MIT license. [Paper](https://arxiv.org/abs/2411.16228) · [Source at that commit](https://github.com/quantumjim/soft_information_decoding/tree/5ef9a62f638b5ee6b276c16f63189b1ce10540a8) · [Data license](../examples/data/LICENSE)

For each prepared state, the example fits a Gaussian to the first 8,000 pairs and mixes in 2% of the other state’s density to limit overconfident predictions. It draws readings for the simulated square from the remaining 7,748 pairs per class, using the synthetic bond labels to select the class. Its log losses measure prediction on that square. Multiple draws can reuse the same hardware measurement.

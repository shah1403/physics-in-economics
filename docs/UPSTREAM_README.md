# Physics in Economics — CDA Monte Carlo Code

Code accompanying *The Physics in Economics* (Shah 2026). Reproduces the
Monte Carlo continuous-double-auction experiment from §6.2 of the paper.

## What this code does

For each of 5 supply–demand environments × 2 initial-condition regimes ×
75 random seeds = 750 simulated CDA runs, it:

1. Runs the audited CDA mechanism (random activation, probabilistic
   acceptance, frustration state, quote improvement) with no leakage of
   the theoretical equilibrium.
2. Detects order-book formation from the live spread series.
3. Smooths the post-formation contract series with a 100-transaction
   moving average within each run.
4. Averages those smoothed paths across 75 seeds to get the cell mean.
5. Fits a one-parameter exponential ODE
   `P_α(β·α) = P* + (P₀ - P*) exp(-β α)`
   where `α = n - n₀` is the post-formation transaction index. The
   convention matches the Vernon Smith fit in §6.1: small β = slow
   per-transaction relaxation.
6. Computes residual statistics, the offset `ΔP₅₀₀`, and the closed-form
   `E[P]` decomposition into geometric and behavioural terms.
7. Produces all main-paper plots.

## Files

| File | Purpose |
|---|---|
| `cda.py` | Audited CDA simulator (`run_cda`, `moving_average`) |
| `environments.py` | 5 supply–demand environments + 2 IC regimes |
| `run_mc.py` | Run the Monte Carlo; writes per-run NPZs + run-level CSV |
| `analyze.py` | Formation detection, β fit, residual stats, `E[P]` decomposition |
| `plot.py` | All main-paper PNG plots |
| `Makefile` | Convenience targets: `make all`, `make quick`, `make clean` |
| `requirements.txt` | Python dependencies |

## Outputs

After `make all` you get, under `results/`:

```
results/
├── runs/                              # raw per-run data (750 .npz files)
├── run_level.csv                      # flat per-run statistics
├── aggregate.csv                      # per-cell aggregates
├── tex_data/
│   ├── fit_summary.json               # per-cell ODE fit (Table 3 source)
│   ├── table3_cell_fit.csv            # Table 3 as CSV
│   ├── EP_decomposition.csv           # E[P] decomposition across 10 cells
│   ├── paths_<env>_<ic>.csv           # individual seed MA paths
│   ├── mean_<env>_<ic>.csv            # cell-mean MA path
│   ├── fit_<env>_<ic>.csv             # fitted ODE curve
│   ├── resids_<env>_<ic>.csv          # per-seed residuals
│   ├── resid_mean_<env>_<ic>.csv      # cell-mean residual
│   └── formation_<env>_<ic>.csv       # mean live-spread vs event index
└── plots/
    ├── fit_<env>.png                  # main combined-IC plot per env
    ├── resids_<env>.png               # residual plot per env
    ├── formation_<env>.png            # formation diagnostic per env
    └── EP_decomposition.png           # 10-cell E[P] vs observed
```

## Quick start

```bash
pip install -r requirements.txt

# Reproduce the full paper experiment (~30-60 min depending on hardware):
make all

# Or do a fast smoke-test (5 seeds, ~2 min):
make quick
```

## Manual usage

```bash
# Run Monte Carlo (default 75 seeds per IC per env)
python run_mc.py --n-seeds 75

# Or subset:
python run_mc.py --n-seeds 10 --envs symmetric asymmetric

# Analyse saved runs (formation, fit, decomposition)
python analyze.py

# Plot
python plot.py
```

## Reproducibility

- Every CDA run is seeded; the same seed gives identical contract
  sequences.
- `run_mc.py` derives seeds as `seed_base + hash(env,ic)%1000 + k` for
  seed-index `k`, so different `--seed-base` values give independent
  Monte Carlo experiments.
- All 750 runs in the paper passed the order-book audit with zero
  crossed-book violations (`crossed_violations` column in `run_level.csv`).

## β convention

The paper uses `t = β·α` so that
`P_α(β·α) = P* + (P₀-P*) exp(-β(n-n₀))`.

In this convention β is a per-transaction *rate*: small β = slow
relaxation. The paper's Vernon Smith fit gives β ≈ 0.194; the CDA
simulations give β in the range 0.0002 – 0.02 depending on the schedule.
The two are directly comparable.

## Citation

If you use this code, please cite the paper:
> A. Shah, *The Physics in Economics* (2026).

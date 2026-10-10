"""
Monte Carlo driver: runs the audited CDA across all 5 environments x 2 ICs x N seeds.

Outputs:
    results/runs/{env}_{ic}_seed{NN}.npz   — per-run contracts + spreads
    results/run_level.csv                   — flat run-level statistics
    results/aggregate.csv                   — per-cell aggregates

Run:
    python run_mc.py --n-seeds 75
    python run_mc.py --n-seeds 5  --envs symmetric asymmetric   # quick test
"""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd

from cda import run_cda
from environments import ENVIRONMENTS, ICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=75,
                        help="Number of seeds per (env, IC) cell (paper: 75)")
    parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS.keys()),
                        help="Subset of environments to run")
    parser.add_argument("--out", default="results",
                        help="Output directory")
    parser.add_argument("--seed-base", type=int, default=10000,
                        help="Base for seed generation")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)

    rows = []
    t_start = time.time()
    total_runs = len(args.envs) * len(ICS) * args.n_seeds
    done = 0

    for env_name in args.envs:
        if env_name not in ENVIRONMENTS:
            print(f"skipping unknown env: {env_name}")
            continue
        env = ENVIRONMENTS[env_name]
        values, costs = env.values_costs()

        for ic_name, ic_params in ICS.items():
            for k in range(args.n_seeds):
                seed = int.from_bytes(hashlib.sha256(f"{args.seed_base}:{env_name}:{ic_name}:{k}".encode()).digest()[:8], "big")
                result = run_cda(values, costs, env.P_star, env.Q_star,
                                 seed=seed, **ic_params)
                contracts = result["contracts"]

                # Save per-run raw data (small after subsampling spreads)
                # Spread series is already sampled every 10 events up to 20k.
                np.savez_compressed(
                    os.path.join(args.out, "runs",
                                 f"{env_name}_{ic_name}_seed{k:02d}.npz"),
                    contracts=contracts.astype(np.float64),
                    book_events=result["book_events"].astype(np.int32),
                    spreads=result["spreads"].astype(np.float64),
                )

                # Run-level summary
                p_star = env.P_star
                n_trades = len(contracts)
                if n_trades >= 500:
                    dp_500 = float(np.mean(contracts[-500:]) - p_star)
                else:
                    dp_500 = np.nan
                rows.append(dict(
                    env=env_name, ic=ic_name, seed_idx=k, seed=seed,
                    phi_b=ic_params["phi_b"], phi_s=ic_params["phi_s"],
                    P_star=p_star, Q_star=env.Q_star,
                    n_trades=n_trades, dp_500=dp_500,
                    crossed_violations=result["crossed_book_violations_sampled"],
                ))
                done += 1
                if done % 25 == 0 or done == total_runs:
                    elapsed = time.time() - t_start
                    rate = done / elapsed
                    eta = (total_runs - done) / rate
                    print(f"  [{done}/{total_runs}] {env_name}/{ic_name}  "
                          f"trades={n_trades}  ΔP_500={dp_500:+.4f}  "
                          f"({rate:.1f} runs/s, ETA {eta:.0f}s)")

    # Save run-level CSV
    df_run = pd.DataFrame(rows)
    df_run.to_csv(os.path.join(args.out, "run_level.csv"), index=False)
    print(f"\nWrote {len(df_run)} run-level rows -> {args.out}/run_level.csv")

    # Aggregate per cell
    grp = df_run.groupby(["env", "ic"])
    agg = grp.agg(
        n_runs=("seed", "count"),
        P_star=("P_star", "first"),
        Q_star=("Q_star", "first"),
        n_trades_mean=("n_trades", "mean"),
        n_trades_min=("n_trades", "min"),
        dp_500_mean=("dp_500", "mean"),
        dp_500_std=("dp_500", "std"),
        crossed_violations_total=("crossed_violations", "sum"),
    ).reset_index()
    agg.to_csv(os.path.join(args.out, "aggregate.csv"), index=False)
    print(f"Wrote per-cell aggregate -> {args.out}/aggregate.csv")
    print(f"\nTotal: {done} runs in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()

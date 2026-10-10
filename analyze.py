"""
Analyse the saved Monte Carlo runs to produce:
    results/tex_data/fit_summary.json      — per-cell ODE fit summary (Table 3)
    results/tex_data/EP_decomposition.csv  — E[P] = geometric + behavioural decomposition
    results/tex_data/{paths,mean,fit,resids,resid_mean,formation}_*.csv  — plot inputs

The fit convention follows the paper's hypothesis t = beta * alpha:
    P_alpha(beta * alpha) = P^* + (P_0 - P^*) exp(-beta * (n - n_0))
where alpha = n - n_0 is the data axis in post-formation transaction units.
Small beta = slow per-transaction relaxation; directly comparable to
Vernon Smith's beta = 0.193583.

Run:
    python analyze.py
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from environments import ENVIRONMENTS, ICS
from cda import moving_average

OUT_DIR = "results"
TEX_DIR = os.path.join(OUT_DIR, "tex_data")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
os.makedirs(TEX_DIR, exist_ok=True)

# Fixed analysis parameters used throughout the paper.
N_0 = 250          # post-formation start (transaction index)
MA_WINDOW = 100    # within-run moving average window
SUBSAMPLE = 25     # subsample stride for plot CSVs


# ---------------- Helpers ----------------

def load_cell_runs(env_name, ic_name):
    """Load all saved seeds for one (env, ic) cell. Returns list of np arrays of contract prices."""
    runs = []
    for fn in sorted(os.listdir(RUNS_DIR)):
        if fn.startswith(f"{env_name}_{ic_name}_seed") and fn.endswith(".npz"):
            data = np.load(os.path.join(RUNS_DIR, fn))
            runs.append(np.asarray(data["contracts"], dtype=float))
    return runs


def cell_mean_path(runs, ma_window=MA_WINDOW):
    """Compute 75-seed mean of within-run moving averages, aligned on common length."""
    ma_per = [moving_average(r, ma_window) for r in runs if len(r) >= ma_window + 10]
    if not ma_per:
        return None, None, None
    min_len = min(len(t) for t, _ in ma_per)
    ma_mat = np.stack([m[:min_len] for _, m in ma_per])
    t_common = ma_per[0][0][:min_len]
    return t_common, ma_mat, ma_mat.mean(axis=0)


def fit_beta(t, P_obs, P_star, P_0, n_0=N_0):
    """
    Fit beta in:  P(t) = P* + (P_0 - P*) * exp(-beta * (t - n_0))
    on the post-formation window t >= n_0. Returns (beta, R^2, RMSE, pred).
    """
    mask = t >= n_0
    x = t[mask].astype(float)
    y = P_obs[mask]
    alpha = x - n_0
    def model(a, beta): return P_star + (P_0 - P_star) * np.exp(-beta * a)
    try:
        popt, _ = curve_fit(model, alpha, y, p0=[0.005],
                            bounds=([1e-8], [10.0]), maxfev=20000)
    except Exception:
        return np.nan, np.nan, np.nan, None, None
    beta = float(popt[0])
    pred = model(alpha, beta)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else np.nan
    signed = float(np.trapezoid(resid, x=x))
    abs_int = float(np.trapezoid(np.abs(resid), x=x))
    rho_bal = abs(signed) / abs_int if abs_int > 0 else np.nan
    return beta, r2, rmse, dict(
        x=x, y=y, alpha=alpha, pred=pred, resid=resid,
        signed_int=signed, abs_int=abs_int, rho_bal=rho_bal,
    ), P_0


# ---------------- Per-cell pipeline ----------------

def analyse_cell(env_name, ic_name, env):
    runs = load_cell_runs(env_name, ic_name)
    if not runs:
        print(f"  no runs for {env_name}/{ic_name}; skipping")
        return None

    t_common, ma_mat, ma_mean = cell_mean_path(runs)
    if t_common is None:
        return None

    P_star = env.P_star
    # P_0 = mean MA value at n_0 (first post-formation index)
    idx0 = np.searchsorted(t_common, N_0)
    P_0 = float(ma_mean[idx0])

    beta, r2, rmse, fit, _ = fit_beta(t_common, ma_mean, P_star, P_0)
    if fit is None:
        return None

    # Mean ΔP_500 across raw seeds
    dp_500 = float(np.mean([np.mean(r[-500:]) - P_star
                            for r in runs if len(r) >= 500]))

    # Write subsampled plot CSVs
    write_plot_csvs(env_name, ic_name, t_common, ma_mat, ma_mean, fit)

    return dict(
        env=env_name, ic=ic_name,
        n_seeds=len(runs), n_0=N_0, ma_window=MA_WINDOW,
        P_star=P_star, P_0=P_0, beta=beta, r2=r2, rmse=rmse,
        dp_500=dp_500, signed_int=fit["signed_int"],
        abs_int=fit["abs_int"], rho_bal=fit["rho_bal"],
    )


def write_plot_csvs(env_name, ic_name, t, ma_mat, ma_mean, fit):
    """Subsampled CSVs used by plot.py."""
    stride = SUBSAMPLE
    sl = slice(0, None, stride)

    # Mean path
    df = pd.DataFrame({"t": t[sl], "Pmean": ma_mean[sl]})
    df.to_csv(os.path.join(TEX_DIR, f"mean_{env_name}_{ic_name}.csv"), index=False)

    # Individual seed MA paths (one column per seed)
    cols = {"t": t[sl]}
    for i in range(ma_mat.shape[0]):
        cols[f"P{i:02d}"] = ma_mat[i, sl]
    pd.DataFrame(cols).to_csv(
        os.path.join(TEX_DIR, f"paths_{env_name}_{ic_name}.csv"), index=False)

    # ODE fit (only post-formation)
    fit_t = fit["x"]; fit_p = fit["pred"]
    df = pd.DataFrame({"t": fit_t[::stride], "Pode": fit_p[::stride]})
    df.to_csv(os.path.join(TEX_DIR, f"fit_{env_name}_{ic_name}.csv"), index=False)

    # Residuals per seed (recompute against fit pred)
    fit_pred_at_t = fit["pred"]
    mask = t >= N_0
    t_post = t[mask]
    # Aligned individual residuals = ma_mat[:, mask] - pred(broadcast)
    resid_mat = ma_mat[:, mask] - fit_pred_at_t[None, :]
    cols = {"t": t_post[::stride]}
    for i in range(resid_mat.shape[0]):
        cols[f"P{i:02d}"] = resid_mat[i, ::stride]
    pd.DataFrame(cols).to_csv(
        os.path.join(TEX_DIR, f"resids_{env_name}_{ic_name}.csv"), index=False)
    # Mean residual
    pd.DataFrame({"t": t_post[::stride],
                  "resid": resid_mat.mean(axis=0)[::stride]}).to_csv(
        os.path.join(TEX_DIR, f"resid_mean_{env_name}_{ic_name}.csv"), index=False)

    # Formation diagnostic: mean live spread across seeds
    # Reload to get spreads
    spread_events_all = []
    spread_values_all = []
    for fn in sorted(os.listdir(RUNS_DIR)):
        if fn.startswith(f"{env_name}_{ic_name}_seed") and fn.endswith(".npz"):
            d = np.load(os.path.join(RUNS_DIR, fn))
            spread_events_all.append(np.asarray(d["book_events"]))
            spread_values_all.append(np.asarray(d["spreads"]))
    if spread_events_all:
        # Bin by event index, average
        max_ev = max(arr.max() for arr in spread_events_all if len(arr) > 0)
        bins = np.arange(0, max_ev + 100, 100)
        means = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)
        for evs, sps in zip(spread_events_all, spread_values_all):
            if len(evs) == 0: continue
            idx = np.searchsorted(bins, evs) - 1
            for i, s in zip(idx, sps):
                if 0 <= i < len(means):
                    means[i] += s; counts[i] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            means = means / np.where(counts > 0, counts, 1)
        centres = 0.5 * (bins[:-1] + bins[1:])
        valid = counts > 0
        pd.DataFrame({"ev": centres[valid], "spread": means[valid]}).to_csv(
            os.path.join(TEX_DIR, f"formation_{env_name}_{ic_name}.csv"), index=False)


# ---------------- E[P] decomposition ----------------

def compute_EP_decomposition(summary):
    """
    Empirical bid/ask + acceptance-probability check from the trading rules:
        E[P] = (p_b * A + p_s * B) / (p_b + p_s)
             = M + (p_b - p_s)/(p_b + p_s) * Delta
    where M = (A+B)/2, Delta = (A-B)/2.

    A, B are measured as the mean best-ask/best-bid in the last 500 contracts
    of each seed. p_b, p_s are the population-mean acceptance probabilities
    over the active pool at that A, B (we use the simple s=1 acceptance form
    1 - A/v and 1 - c/B, since the empirical mean s is ~1.00-1.05 in every cell).
    """
    rows = []
    for env_name in ENVIRONMENTS:
        env = ENVIRONMENTS[env_name]
        values, costs = env.values_costs()
        for ic_name in ICS:
            # Find the cell in summary
            row = next((s for s in summary if s["env"] == env_name and s["ic"] == ic_name), None)
            if row is None: continue
            # Measure A, B from saved runs: average over last 500 contracts of each seed
            A_list = []; B_list = []
            for fn in sorted(os.listdir(RUNS_DIR)):
                if not fn.startswith(f"{env_name}_{ic_name}_seed"): continue
                d = np.load(os.path.join(RUNS_DIR, fn))
                c_arr = np.asarray(d["contracts"], dtype=float)
                if len(c_arr) < 500: continue
                # Approximate A and B from contract distribution in the tail:
                # contracts execute at standing bid or ask, so the high tail of the
                # last 500 contracts is dominated by A and low tail by B.
                tail = c_arr[-500:]
                A_list.append(float(np.percentile(tail, 90)))
                B_list.append(float(np.percentile(tail, 10)))
            if not A_list: continue
            A_mean = float(np.mean(A_list)); B_mean = float(np.mean(B_list))
            # Population acceptance with s=1
            p_b = float(np.mean(np.maximum(0.0, 1.0 - A_mean / np.maximum(values, 1e-12))))
            p_s = float(np.mean(np.maximum(0.0, 1.0 - costs / max(B_mean, 1e-12))))
            denom = p_b + p_s
            if denom == 0:
                continue
            EP = (p_b * A_mean + p_s * B_mean) / denom
            M = 0.5 * (A_mean + B_mean)
            Delta = 0.5 * (A_mean - B_mean)
            geom = M - env.P_star
            behav = (p_b - p_s) / denom * Delta
            rows.append(dict(
                env=env_name, ic=ic_name, P_star=env.P_star,
                A=A_mean, B=B_mean, p_b=p_b, p_s=p_s,
                geometric=geom, behavioural=behav,
                EP_minus_Pstar=EP - env.P_star,
                observed_dp_500=row["dp_500"],
            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TEX_DIR, "EP_decomposition.csv"), index=False)
    print(f"\nWrote E[P] decomposition -> {TEX_DIR}/EP_decomposition.csv")
    return df


# ---------------- Main ----------------

def main():
    summary = []
    for env_name, env in ENVIRONMENTS.items():
        for ic_name in ICS:
            print(f"Analysing {env_name}/{ic_name}...")
            row = analyse_cell(env_name, ic_name, env)
            if row is not None:
                summary.append(row)
                print(f"  beta={row['beta']:.5f}  R^2={row['r2']:+.4f}  "
                      f"RMSE={row['rmse']:.4f}  ΔP_500={row['dp_500']:+.4f}")

    # Write summary
    with open(os.path.join(TEX_DIR, "fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote fit summary -> {TEX_DIR}/fit_summary.json")

    # Build Table 3 (cell fit) as CSV for easy inspection
    df_t3 = pd.DataFrame(summary)[
        ["env", "ic", "P_star", "P_0", "beta", "r2", "rmse", "dp_500", "rho_bal"]
    ]
    df_t3.to_csv(os.path.join(TEX_DIR, "table3_cell_fit.csv"), index=False)
    print(f"Wrote Table 3 -> {TEX_DIR}/table3_cell_fit.csv")

    # E[P] decomposition
    df_ep = compute_EP_decomposition(summary)
    print(df_ep.to_string(index=False))


if __name__ == "__main__":
    main()

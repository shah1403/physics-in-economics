"""
Generate all main-paper figures from the CSVs written by analyze.py.

Outputs PNG files under results/plots/. Each plot has:
  - faint individual seed MA paths
  - bold within-regime mean
  - dashed exponential ODE fit
  - dotted P* line
plus a separate residual figure per env, plus formation diagnostics.

Run:
    python plot.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environments import ENVIRONMENTS

TEX_DIR = "results/tex_data"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _load(name):
    fn = os.path.join(TEX_DIR, name)
    return pd.read_csv(fn) if os.path.exists(fn) else None


def plot_env(env_name, ylim=None, xlim=None):
    """Combined shortage+surplus on the same axes (matches paper layout)."""
    env = ENVIRONMENTS[env_name]
    P_star = env.P_star

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_any = False
    for ic, color, lighter in [("shortage", "steelblue", "lightsteelblue"),
                                ("surplus", "darkorange", "navajowhite")]:
        paths = _load(f"paths_{env_name}_{ic}.csv")
        mean  = _load(f"mean_{env_name}_{ic}.csv")
        fit   = _load(f"fit_{env_name}_{ic}.csv")
        if paths is None or mean is None: continue
        plotted_any = True
        t = paths["t"].values
        for c in paths.columns:
            if c == "t": continue
            ax.plot(t, paths[c].values, color=color, alpha=0.08, lw=0.4)
        ax.plot(mean["t"], mean["Pmean"], color=color, lw=2.0,
                label=f"{ic} mean")
        if fit is not None:
            ax.plot(fit["t"], fit["Pode"], color="red" if ic == "shortage" else "darkred",
                    ls="--", lw=1.5, label=f"{ic} ODE fit")

    if not plotted_any:
        plt.close(fig); return
    ax.axhline(P_star, color="black", ls=":", lw=1, label=f"$P^*={P_star:.3f}$")
    ax.set_xlabel("Transaction index $t$")
    ax.set_ylabel("Price")
    ax.set_title(f"CDA paths and ODE fit — {env_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"fit_{env_name}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_residuals(env_name):
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted_any = False
    for ic, color in [("shortage", "steelblue"), ("surplus", "darkorange")]:
        df = _load(f"resids_{env_name}_{ic}.csv")
        dfm = _load(f"resid_mean_{env_name}_{ic}.csv")
        if df is None or dfm is None: continue
        plotted_any = True
        t = df["t"].values
        for c in df.columns:
            if c == "t": continue
            ax.plot(t, df[c].values, color=color, alpha=0.08, lw=0.4)
        ax.plot(dfm["t"], dfm["resid"], color=color, lw=2.0, label=f"{ic} mean residual")
    if not plotted_any:
        plt.close(fig); return
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Transaction index $t$")
    ax.set_ylabel("Residual $e_n$")
    ax.set_title(f"Residuals against ODE fit — {env_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"resids_{env_name}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_formation(env_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted_any = False
    for ic, color in [("shortage", "steelblue"), ("surplus", "darkorange")]:
        df = _load(f"formation_{env_name}_{ic}.csv")
        if df is None or len(df) == 0: continue
        plotted_any = True
        ax.plot(df["ev"], df["spread"], color=color, lw=1.6, label=f"{ic}")
    if not plotted_any:
        plt.close(fig); return
    ax.set_xlabel("Event index")
    ax.set_ylabel("Mean live spread")
    ax.set_title(f"Order-book formation — {env_name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20000)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"formation_{env_name}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_EP_decomposition():
    df = _load("EP_decomposition.csv")
    if df is None: return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["EP_minus_Pstar"], df["observed_dp_500"],
               s=70, edgecolor="black", facecolor="steelblue", alpha=0.85, zorder=3)
    lo = min(df["EP_minus_Pstar"].min(), df["observed_dp_500"].min())
    hi = max(df["EP_minus_Pstar"].max(), df["observed_dp_500"].max())
    pad = 0.1 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", lw=1, ls="--", zorder=1)
    for _, r in df.iterrows():
        ax.annotate(f"{r['env'][:4]}.{r['ic'][:1]}",
                    (r["EP_minus_Pstar"], r["observed_dp_500"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Analytical $E[P]-P^*$")
    ax.set_ylabel("Observed $\\overline{\\Delta P}_{500}$")
    ax.set_title("$E[P]$ decomposition vs observed offset (10 cells)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "EP_decomposition.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    # Plot windows roughly matching the paper's xlim/ylim
    plot_specs = {
        "symmetric":   dict(ylim=(1.85, 2.40), xlim=(0, 5000)),
        "asymmetric":  dict(ylim=(1.50, 2.05), xlim=(0, 5000)),
        "flat_supply": dict(ylim=(1.45, 2.60), xlim=(0, 3500)),
        "flat_demand": dict(ylim=(1.50, 2.85), xlim=(0, 4600)),
        "nonlinear":   dict(ylim=(1.70, 2.65), xlim=(0, 4000)),
    }
    print("Plotting fit panels...")
    for env_name, spec in plot_specs.items():
        plot_env(env_name, **spec)
    print("Plotting residuals...")
    for env_name in plot_specs:
        plot_residuals(env_name)
    print("Plotting formation diagnostics...")
    for env_name in plot_specs:
        plot_formation(env_name)
    print("Plotting E[P] decomposition...")
    plot_EP_decomposition()
    print(f"\nAll plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()

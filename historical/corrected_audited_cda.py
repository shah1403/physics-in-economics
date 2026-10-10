
"""
Corrected / Audited Continuous Double Auction (CDA)

This script implements the Smith-style CDA version we debugged.

Institution:
    - Empty order book.
    - Random active trader acts each event.
    - Buyer can accept best ask or post/improve bid.
    - Seller can accept best bid or post/improve ask.
    - If a buyer's bid crosses the best ask, trade executes at the standing ask.
    - If a seller's ask crosses the best bid, trade executes at the standing bid.
    - No crossed book is allowed to persist.
    - Trades are actual contract prices, not Walrasian clearing prices.

Agents:
    - Buyers have private values v_i from demand schedule.
    - Sellers have private costs c_j from supply schedule.
    - Agents do NOT know P_star.
    - P_star is used only after simulation for diagnostics/fitting.

Diagnostics:
    - Runs 5 initial conditions.
    - Detects order-book formation using live bid-ask spread, not P_star.
    - Fits ODE after book formation:
          P(k) = P* + (P0 - P*) exp(-(k-k0)/beta)
      to the 100-contract moving average.
    - Computes residuals, MSE, AR(1), and signed residual integral.
    - Generates plots and JSON summary.
"""

import json
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# 1. MARKET PARAMETERS
# ============================================================

SUPPLY_INT = 0.20
SUPPLY_SLOPE = 0.00032

DEMAND_INT = 4.00
DEMAND_SLOPE = 0.00048

N = 5000
MAX_EVENTS = 70000

Q_STAR = (DEMAND_INT - SUPPLY_INT) / (SUPPLY_SLOPE + DEMAND_SLOPE)
P_STAR = SUPPLY_INT + SUPPLY_SLOPE * Q_STAR

BUYER_VALUES = DEMAND_INT - DEMAND_SLOPE * (np.arange(N) + 0.5)
SELLER_COSTS = SUPPLY_INT + SUPPLY_SLOPE * (np.arange(N) + 0.5)


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)

    if len(x) < window:
        return np.array([]), np.array([])

    idx = np.arange(window - 1, len(x))
    ma = np.convolve(x, np.ones(window) / window, mode="valid")

    return idx, ma


def ar1_coefficient(residuals):
    """
    OLS AR(1) on residuals:

        e_t = c + phi e_(t-1) + innovation

    Returns:
        phi, intercept, innovation_mse
    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]

    if len(e) < 3:
        return np.nan, np.nan, np.nan

    y = e[1:]
    x = e[:-1]

    x_mean = x.mean()
    y_mean = y.mean()

    denom = np.sum((x - x_mean) ** 2)

    if denom <= 0:
        return np.nan, np.nan, np.nan

    phi = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - phi * x_mean)

    innovations = y - (intercept + phi * x)
    innovation_mse = float(np.mean(innovations ** 2))

    return phi, intercept, innovation_mse


# ============================================================
# 3. CORRECTED CDA SIMULATION
# ============================================================

def run_cda(
    seed=0,
    alpha_b=0.50,
    alpha_s=1.05,
    delta0=0.01,
    s_max=4.0,
    step_mult=1.5 ** (1 / 5),
    max_events=MAX_EVENTS,
):
    """
    Run one corrected CDA simulation.

    alpha_b:
        Initial buyer bid fraction when book has no bid:
            B_i = alpha_b * v_i

    alpha_s:
        Initial seller ask multiple when book has no ask:
            A_j = alpha_s * c_j

    delta0:
        Base tick improvement for best bid / best ask.

    step_mult:
        Mild aggressiveness increase when trader cannot improve quote.

    Correct CDA mechanics:
        - Bid crossing ask executes at ask.
        - Ask crossing bid executes at bid.
        - Crossed book does not persist.
    """

    rng = np.random.default_rng(seed)

    active_b = np.ones(N, dtype=bool)
    active_s = np.ones(N, dtype=bool)

    # O(1) random active selection through swap-delete arrays/lists
    b_ids = list(range(N))
    s_ids = list(range(N))

    b_pos = np.arange(N)
    s_pos = np.arange(N)

    # Standing quotes
    bids = np.full(N, -np.inf)
    asks = np.full(N, np.inf)

    # Simple aggressiveness / quote-improvement factors
    s_b = np.ones(N)
    s_s = np.ones(N)

    # Heaps for best bid and best ask
    # bid_heap stores (-bid, buyer_id)
    # ask_heap stores (ask, seller_id)
    bid_heap = []
    ask_heap = []

    contracts = []
    contract_events = []
    trade_kinds = []

    buyer_values_traded = []
    seller_costs_traded = []

    # Book diagnostics sampled every 10 events
    book_events = []
    best_bids = []
    best_asks = []
    spreads = []

    violations = []

    no_trade_counter = 0

    # ------------------------------------------------------------
    # Active trader removal
    # ------------------------------------------------------------

    def remove_buyer(i):
        if not active_b[i]:
            return

        active_b[i] = False

        pos = int(b_pos[i])
        last = b_ids[-1]

        b_ids[pos] = last
        b_pos[last] = pos
        b_ids.pop()

        bids[i] = -np.inf

    def remove_seller(j):
        if not active_s[j]:
            return

        active_s[j] = False

        pos = int(s_pos[j])
        last = s_ids[-1]

        s_ids[pos] = last
        s_pos[last] = pos
        s_ids.pop()

        asks[j] = np.inf

    # ------------------------------------------------------------
    # Best quote retrieval with stale-heap cleanup
    # ------------------------------------------------------------

    def best_bid():
        while bid_heap:
            negp, i = bid_heap[0]
            p = -negp

            if active_b[i] and np.isfinite(bids[i]) and abs(bids[i] - p) < 1e-12:
                return p, i

            heapq.heappop(bid_heap)

        return None, None

    def best_ask():
        while ask_heap:
            p, j = ask_heap[0]

            if active_s[j] and np.isfinite(asks[j]) and abs(asks[j] - p) < 1e-12:
                return p, j

            heapq.heappop(ask_heap)

        return None, None

    def post_bid(i, price):
        bids[i] = float(price)
        heapq.heappush(bid_heap, (-float(price), int(i)))

    def post_ask(j, price):
        asks[j] = float(price)
        heapq.heappush(ask_heap, (float(price), int(j)))

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def sample_book(event):
        """
        Sample book every 10 events and audit that book is not crossed.
        """
        if event % 10 != 0 or event > 20000:
            return

        bb, _ = best_bid()
        aa, _ = best_ask()

        if bb is not None and aa is not None and bb >= aa + 1e-10:
            violations.append({
                "event": int(event),
                "type": "crossed_book",
                "best_bid": float(bb),
                "best_ask": float(aa),
            })

        book_events.append(int(event))
        best_bids.append(bb if bb is not None else np.nan)
        best_asks.append(aa if aa is not None else np.nan)
        spreads.append((aa - bb) if (bb is not None and aa is not None) else np.nan)

    # ------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------

    def execute_at_ask(buyer_i, event, kind):
        """
        Buyer accepts standing ask.
        Contract price = best ask.
        """
        aa, seller_j = best_ask()

        if aa is None:
            violations.append({"event": int(event), "type": "no_ask_for_trade"})
            return False

        price = float(aa)

        # Rationality audits
        if price > BUYER_VALUES[buyer_i] + 1e-10:
            violations.append({
                "event": int(event),
                "type": "buyer_negative_surplus",
                "price": price,
                "buyer_value": float(BUYER_VALUES[buyer_i]),
            })

        if price < SELLER_COSTS[seller_j] - 1e-10:
            violations.append({
                "event": int(event),
                "type": "seller_negative_surplus",
                "price": price,
                "seller_cost": float(SELLER_COSTS[seller_j]),
            })

        contracts.append(price)
        contract_events.append(int(event))
        trade_kinds.append(kind)

        buyer_values_traded.append(float(BUYER_VALUES[buyer_i]))
        seller_costs_traded.append(float(SELLER_COSTS[seller_j]))

        remove_buyer(buyer_i)
        remove_seller(seller_j)

        return True

    def execute_at_bid(seller_j, event, kind):
        """
        Seller accepts standing bid.
        Contract price = best bid.
        """
        bb, buyer_i = best_bid()

        if bb is None:
            violations.append({"event": int(event), "type": "no_bid_for_trade"})
            return False

        price = float(bb)

        # Rationality audits
        if price > BUYER_VALUES[buyer_i] + 1e-10:
            violations.append({
                "event": int(event),
                "type": "buyer_negative_surplus",
                "price": price,
                "buyer_value": float(BUYER_VALUES[buyer_i]),
            })

        if price < SELLER_COSTS[seller_j] - 1e-10:
            violations.append({
                "event": int(event),
                "type": "seller_negative_surplus",
                "price": price,
                "seller_cost": float(SELLER_COSTS[seller_j]),
            })

        contracts.append(price)
        contract_events.append(int(event))
        trade_kinds.append(kind)

        buyer_values_traded.append(float(BUYER_VALUES[buyer_i]))
        seller_costs_traded.append(float(SELLER_COSTS[seller_j]))

        remove_seller(seller_j)
        remove_buyer(buyer_i)

        return True

    # ------------------------------------------------------------
    # Main CDA event loop
    # ------------------------------------------------------------

    for event in range(max_events):
        if not b_ids or not s_ids:
            break

        # Stop after most feasible volume has traded and market has gone quiet.
        # Prevents misleading flat event-time book tails.
        if len(contracts) > int(0.90 * Q_STAR) and no_trade_counter > 20000:
            break

        traded = False

        # --------------------------------------------------------
        # Buyer acts
        # --------------------------------------------------------
        if rng.random() < 0.5:
            i = b_ids[int(rng.integers(len(b_ids)))]

            aa, _ = best_ask()

            # Buyer may accept standing ask if profitable.
            if aa is not None and aa <= BUYER_VALUES[i]:
                p_acc = min(1.0, s_b[i] * max(0.0, 1.0 - aa / BUYER_VALUES[i]))

                if rng.random() < p_acc:
                    traded = execute_at_ask(i, event, "buyer_accepts_ask")

            # If no acceptance, buyer posts/improves bid.
            if not traded:
                bb, _ = best_bid()

                if bb is None:
                    new_bid = alpha_b * BUYER_VALUES[i]
                else:
                    new_bid = bb + delta0 * s_b[i]

                new_bid = min(new_bid, BUYER_VALUES[i])

                # Correct CDA rule:
                # If new bid crosses best ask, it executes at standing ask.
                aa, _ = best_ask()

                if aa is not None and new_bid >= aa and aa <= BUYER_VALUES[i]:
                    traded = execute_at_ask(i, event, "buyer_crosses_ask")

                else:
                    bb, _ = best_bid()

                    if bb is None or new_bid > bb:
                        post_bid(i, new_bid)
                    else:
                        s_b[i] = min(s_max, s_b[i] * step_mult)

        # --------------------------------------------------------
        # Seller acts
        # --------------------------------------------------------
        else:
            j = s_ids[int(rng.integers(len(s_ids)))]

            bb, _ = best_bid()

            # Seller may accept standing bid if profitable.
            if bb is not None and bb >= SELLER_COSTS[j]:
                p_acc = min(1.0, s_s[j] * max(0.0, 1.0 - SELLER_COSTS[j] / max(bb, 1e-12)))

                if rng.random() < p_acc:
                    traded = execute_at_bid(j, event, "seller_accepts_bid")

            # If no acceptance, seller posts/improves ask.
            if not traded:
                aa, _ = best_ask()

                if aa is None:
                    new_ask = alpha_s * SELLER_COSTS[j]
                else:
                    new_ask = aa - delta0 * s_s[j]

                new_ask = max(new_ask, SELLER_COSTS[j])

                # Correct CDA rule:
                # If new ask crosses best bid, it executes at standing bid.
                bb, _ = best_bid()

                if bb is not None and new_ask <= bb and bb >= SELLER_COSTS[j]:
                    traded = execute_at_bid(j, event, "seller_crosses_bid")

                else:
                    aa, _ = best_ask()

                    if aa is None or new_ask < aa:
                        post_ask(j, new_ask)
                    else:
                        s_s[j] = min(s_max, s_s[j] * step_mult)

        no_trade_counter = 0 if traded else no_trade_counter + 1
        sample_book(event)

    return {
        "contracts": np.array(contracts, dtype=float),
        "contract_events": np.array(contract_events, dtype=int),
        "trade_kinds": trade_kinds,
        "buyer_values_traded": np.array(buyer_values_traded, dtype=float),
        "seller_costs_traded": np.array(seller_costs_traded, dtype=float),
        "book_events": np.array(book_events, dtype=int),
        "best_bids": np.array(best_bids, dtype=float),
        "best_asks": np.array(best_asks, dtype=float),
        "spreads": np.array(spreads, dtype=float),
        "remaining_buyers": len(b_ids),
        "remaining_sellers": len(s_ids),
        "violations": violations,
        "last_event": int(event),
        "alpha_b": alpha_b,
        "alpha_s": alpha_s,
        "seed": seed,
    }


# ============================================================
# 4. ORDER-BOOK FORMATION DETECTION
# ============================================================

def detect_book_formation_index(result, ma_window=100):
    """
    Detect book formation using the live bid-ask spread only.

    Rule:
        - Sample spread S_t = best_ask - best_bid.
        - Compute 30-sample rolling median.
        - Formation event is first time rolling median spread < 0.08.
        - Map event to contract index.
        - Enforce at least 250-contract burn-in.

    This does NOT use P_star.
    """

    ev = result["book_events"]
    sp = result["spreads"]

    finite = np.isfinite(sp)
    formation_event = None

    if finite.sum() >= 40:
        ev_f = ev[finite]
        sp_f = sp[finite]

        w = 30
        rolling_median = np.array([
            np.median(sp_f[i:i + w])
            for i in range(len(sp_f) - w + 1)
        ])

        candidates = np.flatnonzero(rolling_median < 0.08)

        if len(candidates):
            formation_event = int(ev_f[candidates[0] + w - 1])

    if formation_event is None:
        formation_contract = min(500, max(0, len(result["contracts"]) - 50))
        formation_event = int(result["contract_events"][formation_contract]) if len(result["contracts"]) else 0

    else:
        ce = result["contract_events"]
        formation_contract = int(np.searchsorted(ce, formation_event, side="left"))
        formation_contract = max(250, formation_contract)

    # Ensure enough moving-average data remains.
    formation_contract = min(
        formation_contract,
        max(0, len(result["contracts"]) - ma_window - 10),
    )

    return formation_contract, formation_event


# ============================================================
# 5. POST-FORMATION ODE FIT + RESIDUAL DIAGNOSTICS
# ============================================================

def fit_post_formation_residuals(result, formation_idx, ma_window=100):
    """
    Fit the ODE envelope to 100-contract moving average after book formation:

        P(k) = P* + (P0 - P*) exp(-(k-k0)/beta)

    Then compute:
        - residuals
        - MSE
        - RMSE
        - MAE
        - AR(1)
        - signed residual integral
    """

    mk, ma = moving_average(result["contracts"], ma_window)

    mask = mk >= formation_idx

    x = mk[mask]
    observed = ma[mask]

    if len(observed) < 20:
        return None

    x0 = int(x[0])
    P0 = float(observed[0])

    def model(xvals, beta):
        return P_STAR + (P0 - P_STAR) * np.exp(-(xvals - x0) / beta)

    popt, _ = curve_fit(
        model,
        x,
        observed,
        p0=[300.0],
        bounds=([1e-6], [20000.0]),
        maxfev=20000,
    )

    beta = float(popt[0])

    predicted = model(x, beta)
    residuals = observed - predicted

    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    phi, intercept, innovation_mse = ar1_coefficient(residuals)

    signed_integral = float(np.trapz(residuals, x=x))
    signed_sum = float(np.sum(residuals))

    length = float(x[-1] - x[0]) if len(x) > 1 else np.nan
    mean_residual_from_integral = float(signed_integral / length) if length > 0 else np.nan

    return {
        "x": x,
        "observed_ma": observed,
        "predicted": predicted,
        "residuals": residuals,
        "beta": beta,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ar1_phi": phi,
        "ar1_intercept": intercept,
        "ar1_innovation_mse": innovation_mse,
        "signed_integral_trapz": signed_integral,
        "signed_sum_residuals": signed_sum,
        "mean_residual_from_integral": mean_residual_from_integral,
        "formation_contract": int(formation_idx),
        "formation_event": int(result["contract_events"][formation_idx]),
        "P0_post": P0,
    }


# ============================================================
# 6. PLOTS
# ============================================================

def make_plots(results, fits):
    """
    Generate diagnostic plots.
    """

    # ------------------------------------------------------------
    # Plot 1: moving averages and ODE fits
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, result in results.items():
        mk, ma = moving_average(result["contracts"], 100)
        ax.plot(mk, ma, linewidth=2.0, label=f"{name} MA")

        fit = fits[name]

        if fit is not None:
            ax.axvline(fit["formation_contract"], linestyle=":", alpha=0.25)
            ax.plot(
                fit["x"],
                fit["predicted"],
                "--",
                linewidth=1.8,
                label=f"{name} ODE beta={fit['beta']:.0f}",
            )

    ax.axhline(P_STAR, linestyle=":", linewidth=2.3, label=f"P*={P_STAR:.3f}")
    ax.set_title("CDA: post-book-formation ODE fits")
    ax.set_xlabel("Contract number")
    ax.set_ylabel("100-contract moving average price")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("cda_post_formation_ode_fits.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: residuals
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, fit in fits.items():
        if fit is None:
            continue

        ax.plot(
            fit["x"],
            fit["residuals"],
            linewidth=1.4,
            label=f"{name} residuals phi={fit['ar1_phi']:.2f}",
        )

    ax.axhline(0, linestyle=":", linewidth=2)
    ax.set_title("Post-formation ODE residuals")
    ax.set_xlabel("Contract number")
    ax.set_ylabel("Residual: observed MA - predicted ODE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("cda_post_formation_residuals.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: signed cumulative integral of residuals
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, fit in fits.items():
        if fit is None:
            continue

        x = fit["x"]
        e = fit["residuals"]

        dx = np.diff(x)
        cumulative = np.zeros_like(e)
        cumulative[1:] = np.cumsum(0.5 * (e[1:] + e[:-1]) * dx)

        ax.plot(
            x,
            cumulative,
            linewidth=2,
            label=f"{name}: final integral={fit['signed_integral_trapz']:.2f}",
        )

    ax.axhline(0, linestyle=":", linewidth=2)
    ax.set_title("Cumulative signed integral of post-formation residuals")
    ax.set_xlabel("Contract number")
    ax.set_ylabel("Signed cumulative integral")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("cda_residual_cumulative_signed_integral.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: MSE bar chart
    # ------------------------------------------------------------
    names = list(fits.keys())
    mse_vals = [
        fits[name]["mse"] if fits[name] is not None else np.nan
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, mse_vals)
    ax.set_title("Post-formation ODE residual MSE")
    ax.set_xlabel("Initial condition")
    ax.set_ylabel("MSE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("cda_post_formation_mse_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot 5: AR(1) phi bar chart
    # ------------------------------------------------------------
    phi_vals = [
        fits[name]["ar1_phi"] if fits[name] is not None else np.nan
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, phi_vals)
    ax.axhline(0, linestyle=":", linewidth=2)
    ax.set_title("AR(1) coefficient of post-formation ODE residuals")
    ax.set_xlabel("Initial condition")
    ax.set_ylabel("AR(1) phi")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("cda_post_formation_ar1_phi_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Plot 6: order-book spread formation diagnostic
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, result in results.items():
        sp = result["spreads"]
        ev = result["book_events"]

        finite = np.isfinite(sp)

        ax.plot(
            ev[finite],
            sp[finite],
            linewidth=1.0,
            alpha=0.75,
            label=name,
        )

    ax.axhline(0.08, linestyle="--", linewidth=1.5, label="formation spread threshold 0.08")
    ax.axhline(0.0, linestyle=":", linewidth=2)
    ax.set_title("CDA order-book formation diagnostic: sampled live spread")
    ax.set_xlabel("Event number")
    ax.set_ylabel("Best ask - best bid")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("cda_spread_formation_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. SUMMARY
# ============================================================

def summarize_run(result, fit):
    contracts = result["contracts"]

    kind_counts = {}
    for kind in result["trade_kinds"]:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    buyer_surplus = result["buyer_values_traded"] - contracts
    seller_surplus = contracts - result["seller_costs_traded"]

    summary = {
        "alpha_b": result["alpha_b"],
        "alpha_s": result["alpha_s"],
        "seed": result["seed"],
        "n_contracts": int(len(contracts)),
        "remaining_buyers": int(result["remaining_buyers"]),
        "remaining_sellers": int(result["remaining_sellers"]),
        "last_event": int(result["last_event"]),
        "violation_count": int(len(result["violations"])),
        "violation_types": sorted(list(set(v["type"] for v in result["violations"]))),
        "trade_kind_counts": kind_counts,
        "mean_first_100": float(np.mean(contracts[:100])) if len(contracts) >= 100 else None,
        "mean_last_100": float(np.mean(contracts[-100:])) if len(contracts) >= 100 else None,
        "mean_last_500": float(np.mean(contracts[-500:])) if len(contracts) >= 500 else None,
        "error_last_500": float(np.mean(contracts[-500:]) - P_STAR) if len(contracts) >= 500 else None,
        "min_buyer_surplus": float(np.min(buyer_surplus)) if len(buyer_surplus) else None,
        "min_seller_surplus": float(np.min(seller_surplus)) if len(seller_surplus) else None,
    }

    if fit is not None:
        summary["post_formation_fit"] = {
            "formation_contract": fit["formation_contract"],
            "formation_event": fit["formation_event"],
            "P0_post": fit["P0_post"],
            "beta": fit["beta"],
            "mse": fit["mse"],
            "rmse": fit["rmse"],
            "mae": fit["mae"],
            "ar1_phi": fit["ar1_phi"],
            "ar1_intercept": fit["ar1_intercept"],
            "ar1_innovation_mse": fit["ar1_innovation_mse"],
            "signed_integral_trapz": fit["signed_integral_trapz"],
            "signed_sum_residuals": fit["signed_sum_residuals"],
            "mean_residual_from_integral": fit["mean_residual_from_integral"],
            "n_residuals": int(len(fit["residuals"])),
        }

    else:
        summary["post_formation_fit"] = None

    return summary


# ============================================================
# 8. MAIN
# ============================================================

def main():
    conditions = [
        ("extreme_shortage", 0.20, 1.02, 701),
        ("shortage", 0.50, 1.05, 702),
        ("mid_mixed", 0.75, 1.25, 703),
        ("surplus", 0.95, 1.50, 704),
        ("extreme_surplus", 1.00, 5.00, 705),
    ]

    results = {}
    fits = {}

    start = time.time()

    for name, alpha_b, alpha_s, seed in conditions:
        result = run_cda(seed=seed, alpha_b=alpha_b, alpha_s=alpha_s)
        formation_idx, _ = detect_book_formation_index(result)
        fit = fit_post_formation_residuals(result, formation_idx)

        results[name] = result
        fits[name] = fit

    runtime = time.time() - start

    make_plots(results, fits)

    summary = {
        "version": "corrected_audited_cda",
        "P_star": P_STAR,
        "Q_star": Q_STAR,
        "N": N,
        "MAX_EVENTS": MAX_EVENTS,
        "runtime_seconds": runtime,
        "mechanics": [
            "empty book",
            "sequential CDA",
            "buyer accepts best ask or posts bid",
            "seller accepts best bid or posts ask",
            "crossing bid executes at standing ask",
            "crossing ask executes at standing bid",
            "no crossed book allowed to persist",
            "agents do not use P_star",
        ],
        "book_formation_rule": "First sampled event where 30-sample rolling median spread < 0.08, mapped to contract index, with minimum 250-contract burn-in. Uses only book state, not P_star.",
        "fit_rule": "Fit ODE to 100-contract moving average after detected book formation.",
        "residual_rule": "residual = observed 100-contract MA - predicted ODE path",
        "runs": {
            name: summarize_run(results[name], fits[name])
            for name in results.keys()
        },
        "plots": {
            "ode_fits": "cda_post_formation_ode_fits.png",
            "residuals": "cda_post_formation_residuals.png",
            "cumulative_signed_integral": "cda_residual_cumulative_signed_integral.png",
            "mse_bar": "cda_post_formation_mse_bar.png",
            "ar1_phi_bar": "cda_post_formation_ar1_phi_bar.png",
            "spread_formation_diagnostic": "cda_spread_formation_diagnostic.png",
        },
    }

    with open("cda_audited_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

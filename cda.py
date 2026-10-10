"""
Audited Continuous Double Auction (CDA) simulator.

A single simulation run is `run_cda(values, costs, p_star, q_star, seed, ...)`.
The mechanism, behavioural layer, quote-posting rules, and acceptance functions
are exactly as described in the paper (§6.2).

The function returns a dict with:
    contracts                       — array of realised contract prices
    book_events                     — event indices at which spread was sampled
    spreads                         — corresponding live spread (A - B)
    crossed_book_violations_sampled — count of sampled events with B >= A
                                      (should be 0 across all runs)

The simulator is deterministic given a seed.
"""
import heapq
import numpy as np


def run_cda(values, costs, p_star, q_star, seed=0,
            phi_b=0.50, phi_s=1.05, delta0=0.01, max_events=70000,
            sample_book_until=20000, sample_book_every=10):
    N = len(values)
    rng = np.random.default_rng(seed)

    active_b = np.ones(N, dtype=bool)
    active_s = np.ones(N, dtype=bool)

    b_ids = list(range(N))
    s_ids = list(range(N))
    b_pos = np.arange(N)
    s_pos = np.arange(N)

    bids = np.full(N, -np.inf)
    asks = np.full(N, np.inf)

    s_b = np.ones(N)  # buyer frustration / aggressiveness
    s_s = np.ones(N)  # seller frustration / aggressiveness

    bid_heap = []
    ask_heap = []

    contracts = []
    contract_events = []

    book_events = []
    spreads = []

    crossed_book_violations = 0
    no_trade_counter = 0
    step_mult = 1.5 ** (1 / 5)

    def remove_buyer(i):
        if not active_b[i]: return
        active_b[i] = False
        pos = int(b_pos[i]); last = b_ids[-1]
        b_ids[pos] = last; b_pos[last] = pos
        b_ids.pop(); bids[i] = -np.inf

    def remove_seller(j):
        if not active_s[j]: return
        active_s[j] = False
        pos = int(s_pos[j]); last = s_ids[-1]
        s_ids[pos] = last; s_pos[last] = pos
        s_ids.pop(); asks[j] = np.inf

    def best_bid():
        while bid_heap:
            negp, i = bid_heap[0]; p = -negp
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

    def sample_book(event):
        nonlocal crossed_book_violations
        if event % sample_book_every != 0 or event > sample_book_until: return
        bb, _ = best_bid(); aa, _ = best_ask()
        if bb is not None and aa is not None:
            if bb >= aa + 1e-10:
                crossed_book_violations += 1
            book_events.append(event); spreads.append(aa - bb)

    def execute_at_ask(buyer_i, event):
        aa, seller_j = best_ask()
        if aa is None: return False
        contracts.append(float(aa)); contract_events.append(int(event))
        remove_buyer(buyer_i); remove_seller(seller_j)
        return True

    def execute_at_bid(seller_j, event):
        bb, buyer_i = best_bid()
        if bb is None: return False
        contracts.append(float(bb)); contract_events.append(int(event))
        remove_seller(seller_j); remove_buyer(buyer_i)
        return True

    for event in range(int(max_events)):
        if not b_ids or not s_ids: break
        if np.isfinite(q_star) and q_star > 0:
            if len(contracts) > int(0.90 * q_star) and no_trade_counter > 20000:
                break

        traded = False
        if rng.random() < 0.5:
            # Buyer selected
            i = b_ids[int(rng.integers(len(b_ids)))]
            aa, _ = best_ask()
            if aa is not None and aa <= values[i]:
                p_acc = min(1.0, s_b[i] * max(0.0, 1.0 - aa / max(values[i], 1e-12)))
                if rng.random() < p_acc:
                    traded = execute_at_ask(i, event)
            if not traded:
                bb, _ = best_bid()
                new_bid = (phi_b * values[i]) if bb is None else (bb + delta0 * s_b[i])
                new_bid = min(new_bid, values[i])
                aa, _ = best_ask()
                if aa is not None and new_bid >= aa and aa <= values[i]:
                    traded = execute_at_ask(i, event)
                else:
                    bb, _ = best_bid()
                    if bb is None or new_bid > bb:
                        post_bid(i, new_bid)
                    else:
                        s_b[i] = min(4.0, s_b[i] * step_mult)
        else:
            # Seller selected
            j = s_ids[int(rng.integers(len(s_ids)))]
            bb, _ = best_bid()
            if bb is not None and bb >= costs[j]:
                p_acc = min(1.0, s_s[j] * max(0.0, 1.0 - costs[j] / max(bb, 1e-12)))
                if rng.random() < p_acc:
                    traded = execute_at_bid(j, event)
            if not traded:
                aa, _ = best_ask()
                new_ask = (phi_s * costs[j]) if aa is None else (aa - delta0 * s_s[j])
                new_ask = max(new_ask, costs[j])
                bb, _ = best_bid()
                if bb is not None and new_ask <= bb and bb >= costs[j]:
                    traded = execute_at_bid(j, event)
                else:
                    aa, _ = best_ask()
                    if aa is None or new_ask < aa:
                        post_ask(j, new_ask)
                    else:
                        s_s[j] = min(4.0, s_s[j] * step_mult)

        no_trade_counter = 0 if traded else no_trade_counter + 1
        sample_book(event)

    return {
        "contracts": np.array(contracts, dtype=float),
        "book_events": np.array(book_events, dtype=int),
        "spreads": np.array(spreads, dtype=float),
        "crossed_book_violations_sampled": int(crossed_book_violations),
    }


def moving_average(x, window=100):
    """Trailing simple moving average. Returns (indices, ma_values)."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return np.array([]), np.array([])
    idx = np.arange(window - 1, len(x))
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    return idx, ma

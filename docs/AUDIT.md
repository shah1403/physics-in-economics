# Verification and model limits

## What was recovered

The supplied 48-page PDF and upstream commit `30bf9e6347f69146b3d1f3aeb7e6f9b0bc5477e3` are retained. The `historical/` CSV contains **75 total runs across three environments and three initial conditions**, not the final paper's 750 runs. These are different experiments and must not be combined.

## Reproduced mathematics

For positive relaxation coefficient, x' = a - b x has solution x* + (x0-x*) exp(-b t), with x*=a/b. The OU extension has the same mean under constant coefficients and integrable initial condition. This is a modeling assumption for market adjustment, not a consequence of static supply/demand curves alone. Perfect information and absence of delays alone do not establish the proposed law.

The Smith Test 1 table contains 28 chart-digitized prices. Refitting P(t)=2-0.3 exp(-t/tau) gives tau approximately 5.16585 and R-squared approximately 0.561401, matching the paper. This is an **in-sample curve fit**. It is not out-of-sample forecasting or a CDA recreation of Smith's human trading experiment.

The paper's plotted `100*abs(predicted-observed)` is an absolute dollar error multiplied by 100, not percentage error. The corrected relative error divides by the observed price. 25/28 observations satisfy a 5% relative-error threshold; 18/28 satisfy the paper's plotted threshold. The supplied PDF is preserved unchanged, with this erratum alongside it.

## Code/documentation discrepancies

1. Upstream Python hash-based seeds change across interpreter processes. SHA-256-derived seeds fix this for new runs. Exact original paper seeds are unavailable; these are new replications.
2. Paper beta is a time constant in exp(-t/beta). `analyze.py` beta is a rate in exp(-beta*t). They are reciprocals: the Smith rate is about 0.193579 per transaction. The original README's statement that these beta values are directly comparable is misleading.
3. `analyze.py` uses a **fixed n0=250**, not detected formation. Plotting average spread is not a formation detector.
4. The E[P] routine uses the 90th/10th percentiles of tail contract prices as ask/bid proxies. It does not observe the contemporaneous best quotes or active trader pool. It averages acceptance probabilities over the initial population and assumes frustration s=1. Its output is an exploratory proxy calculation, not a measured mechanism decomposition or verification of the paper's R-squared=0.998 claim.
5. `p_star` is unused in trading decisions, but `q_star` enters a stopping condition. Therefore the entire simulation cannot be described as having no access to equilibrium information.
6. Crossed-book checks sample every 10 events only through event 20,000. Zero violations means zero **sampled** violations; it does not certify every event or each trader's realized surplus.
7. NumPy 2.0 is required by `np.trapezoid`; upstream allowed NumPy 1.22. Raw float32 storage also introduced avoidable rounding; new runs retain float64.
8. Smoothed cell-mean fit R-squared is not individual-transaction predictive R-squared. Overlapping moving averages are dependent. A fixed-equilibrium exponential can fit poorly or have negative R-squared; retain these cases.

## Scope

The rerun covers all five environments and both initial conditions at 75 seeds per cell. Source and input hashes, run-level outputs, parameter fits and command logs are retained. It is a fresh numerical replication with audit corrections, not proof of universal market convergence, exact original-seed replication, or validation of every theoretical claim. The original PDF and legacy dashboard remain historical documents.

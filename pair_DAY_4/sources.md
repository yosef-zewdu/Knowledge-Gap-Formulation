# Sources — Day 4

## Canonical Papers / Docs

**1. Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.**
- Chapters 12–14 cover the percentile method CI: when it is consistent, when it undercovers, and what sample size assumptions underpin the guarantee.
- Foundational reference for why `bootstrap_ci()` uses the 2.5th/97.5th percentile of the bootstrap distribution rather than a normal approximation, and for understanding the finite-variance assumption that makes continuous bounded scores safe to bootstrap.

**2. Dror, R., Baumer, G., Shlain, S., & Reichart, R. (2018). "Deep Dominance — How to Properly Compare Deep Neural Models." *Proceedings of ACL 2018*.**
- Empirically establishes that bootstrap CI coverage in NLP evaluation is unreliable below n ≈ 300 — directly answering whether n=60 is enough to trust a stated 95% interval (it is not; true coverage is closer to 88–92%).
- Shows that permutation tests are better calibrated than percentile bootstrap for significance decisions at typical NLP eval set sizes, which is why the permutation p-value (0.0153) is the stronger evidence here than the CI lower bound alone.
- URL: https://aclanthology.org/P18-1128/

## Tool / Pattern Used Hands-On

**`bootstrap_ci()` and `paired_sign_flip_p_value()` in `ablations/run_ablation_analysis.py` — ruthasolll/tenacious-sales-agent-evaluation-bench**
- Reading both functions side by side on the actual 60-task dev-set delta values (18 improved / 6 regressed / 36 unchanged) makes the spike-at-zero distribution concrete and demonstrates why the two tests answer different questions: the permutation test asks "is the gap real?" and the bootstrap CI asks "how wide is our uncertainty about the gap's magnitude?"
- The CI half-width sanity check (`SD(deltas) / sqrt(60)` compared against the observed half-width of ≈ 2.6 points) can be run directly in Python using `statistics.stdev()` on the per-task delta values from `outputs/evaluation_comparison.json`.
- The Delta A / Delta B framing in `final_training_report.md` is also the clearest hands-on demonstration of why Delta B describes baseline strength rather than a separately bootstrapped number — reading that report alongside the ablation script closes the gap between what the numbers say and what they mean.

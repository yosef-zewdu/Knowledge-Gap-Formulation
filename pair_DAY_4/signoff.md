# Asker Sign-Off — Day 4

**Asker:** Yosef Zewdu
**Gap status:** Closed

---

Ruth's explainer closed the central part of my question: I now have a clear mechanical account of why `score_bench.py`'s current approach loses exactly the information I needed — the difference between an agent that can solve a task sometimes and one that never solves it. The key move was naming the specific information collapse: averaging pass@1 across five independent runs discards the query-level retry structure, so a task the agent solved once in five attempts looks nearly identical to a task it never solved. Before reading the explainer, I understood that five runs existed in the benchmark; I did not have a model for why aggregating them independently was a category error rather than just a slightly imprecise metric.

The pass@k formula and the Query A / Query B worked example were the most immediately useful parts. Seeing pass@5 = 1.0 for a query with one success out of five runs — compared to pass@5 = 0 for a query with zero successes — made the distinction between stochastic instability and capability gap concrete rather than abstract. The consequence I hadn't articulated before: the two failure types need completely different fixes. Retries, temperature tuning, and self-consistency address instability. Retrieval improvement, prompt redesign, and capability training address gaps. Averaging pass@1 makes both look like the same problem requiring the same intervention.

The implementation section also moved my understanding. The `pass_at_k(n, c, k)` function using `comb(n-c, k) / comb(n, k)` is the unbiased estimator from the Codex paper, and it operates on grouped query-level results rather than the run-level aggregation my current script does. That is a concrete, testable edit to `score_bench.py` — not a theoretical improvement.

What I now understand that I did not before: pass@k is not a more sophisticated version of pass@1. It answers a different question entirely. pass@1 measures single-attempt reliability; pass@k measures whether the capability exists anywhere in the model's output distribution. For stochastic agents evaluated across multiple runs, pass@k is the appropriate primary metric, and reporting only per-run pass@1 averages is an evaluation design error.

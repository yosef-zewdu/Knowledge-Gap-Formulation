# Grounding Commit — Day 4

**Author:** Yosef Zewdu

---

## What was edited

**File:** `scripts/score_bench.py` in the Week 8–9 data-agent-challenge repo  

## The edit

Added a query-level pass@k computation block after the existing per-run aggregation. The block groups results by query ID across all five `score.json` files, then computes pass@1, pass@3, and pass@5 using the unbiased estimator from Chen et al. (2021), and prints a breakdown separating capability gaps (pass@5 = 0) from high-variance misses (pass@5 > 0, pass@1 < 0.4).


## Why this grounds the gap

Before this edit, the script had no mechanism to distinguish between a query the agent never solved and one it solved inconsistently. Both produced low per-run pass@1 averages and were treated identically by the aggregation logic. This edit makes the distinction visible at evaluation time: capability gaps surface as queries where pass@5 = 0 across all five runs, while high-variance misses surface as queries where pass@5 > 0 but pass@1 is low — meaning the capability exists somewhere in the agent's output distribution but is not reliably expressed.

The edit does not change how existing runs are collected. It adds a second reporting layer on top of the current aggregation so both metrics are visible simultaneously.

## Connection to the week's gap

The morning call identified that the operational risk in my benchmark was treating stochastic instability and capability absence as the same failure mode. This commit makes the difference non-silent. Any FDE running this script now gets a direct breakdown of which queries need capability-level intervention (retrieval, reasoning, training) versus which need reliability-level intervention (temperature, retries, self-consistency) — the two categories the explainer showed require completely different engineering responses.

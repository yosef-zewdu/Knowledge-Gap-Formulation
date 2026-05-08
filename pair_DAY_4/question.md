# Day 4 Question — pass@k vs pass@1

**Asker:** Yosef Zewdu 
---

## Gap Question

 `score_bench.py` collapses five runs per query into five separate pass@1 numbers, making a query the agent solved once indistinguishable from one it never solved. What does pass@k estimator capture that the per-run average does not — and how would computing it over our five `score.json` files per query reveal which failures are high-variance misses vs. consistent capability gaps, changing what we should try next to improve the agent?

---

## Grounding Artifact

**File:** `scripts/score_bench.py` in the Week 8–9 data-agent-challenge repo
([PALM-Oracle-Forge/data-agent-challenge](https://github.com/PALM-Oracle-Forge/data-agent-challenge/blob/main/scripts/score_bench.py))

---


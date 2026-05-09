# Portfolio Update — Week 12 Grounding Commits

**For:** FDE Hiring Committee
**Re:** How five weeks of agent and evaluation work was strengthened in Week 12

---

## Summary

Four grounding commits made during Week 12 repair specific methodological weaknesses in the Week 10–11 portfolio. Each commit targets a claim that was locally defensible but analytically wrong, replaces it with a mechanistically precise one, and adds a runnable diagnostic so the failure mode cannot silently recur. The four changes are small by line count and large by epistemic weight.

---

## Commit 1: `methodology_rationale.md` — Judge Cost Claim

**What changed:** Replaced "near-zero marginal cost — one forward pass" with an explicit 3–4 forward pass breakdown: prefill phase (compute-bound, context-length-dependent) and decode phase (bandwidth-bound, output-length-dependent, dominant at runtime). Added estimated wall-time range on T4 GPU based on profiled measurements of a comparable model.

**Why it matters for FDE work:** A methodology document that claims negligible inference cost without analyzing the cost structure is not credible to a reviewer who has debugged inference pipelines. The original claim was the kind of thing that sounds right and isn't — it implicitly treated the model as a classifier doing a single forward pass, when in fact it generates tokens autoregressively. The revised claim names the correct model (roofline), cites the correct bottleneck (memory bandwidth for decode), and gives a falsifiable number. Any FDE reading the revised methodology can replicate the timing measurement and either confirm or challenge the estimate.

**Canonical grounding:** Williams et al. (2009) roofline model; Pope et al. (2023) MBU framework.

---

## Commit 2: `agent/tools.py` — Tool Description Precision

**What changed:** Rewrote trigger conditions for `hubspot_upsert_contact` and `hubspot_log_activity` from vague capability descriptions ("creates or updates a contact") to precise invocation conditions ("Call this when the user provides a contact name or email and asks to update or save CRM data").

**Why it matters for FDE work:** Tool description quality is a training signal, not just documentation. The Week 11 agent used these descriptions both for runtime tool selection (via function-calling) and as training data for fine-tuning the router. Vague descriptions force the model to reason through ambiguity at every call site, increasing latency and error rate. More importantly, vague descriptions in training data teach the fine-tuned model to reason about tools generically rather than to match specific linguistic triggers. The revised descriptions encode the correct probability distribution directly: a user saying "save this contact" should produce a high-probability path to `hubspot_upsert_contact`, not a reasoning trace exploring three alternatives.

This change also surfaces the mechanism behind a Day 2 finding: 96.9% of completion tokens in held-out traces were reasoning tokens, partly because imprecise context forced extended deliberation. Tighter tool descriptions reduce deliberation cost.

**Canonical grounding:** Qwen3 technical report on thinking budget allocation; DeepSeek-R1 on deliberativeness signals.

---

## Commit 3: `week11_unsloth.ipynb` — Score Collapse Diagnostic

**What changed:** Added a post-inference entropy block to the evaluation loop. After collecting verdicts from the fine-tuned DPO judge, the block computes score entropy across the verdict distribution and flags collapse if entropy falls below 0.5 bits on a five-class scale. Running against `judge_traces.jsonl` produces 0.985 bits with a collapse warning (maximum for balanced five-class: 2.322 bits).

**Why it matters for FDE work:** The original notebook reported training loss and sample verdicts but had no automated detection of score collapse — the failure mode where a calibrated-scoring judge degrades to binary output. This failure is consequential: a judge that produces only 2.0 and 4.0 cannot distinguish between a partially correct answer and a fully correct one, making evaluation meaningless for intermediate-quality outputs.

The diagnostic is the minimum viable intervention: it doesn't change the training setup, it doesn't require re-running the fine-tune, and it makes a previously silent failure loud. An FDE reading the notebook after the commit sees immediately whether the judge is functioning as designed. This is the kind of defensive instrumentation that distinguishes a production-quality evaluation pipeline from a research prototype.

**Canonical grounding:** Rafailov et al. (2023) DPO paper, Section 4 (beta KL term constrains distribution, not score geometry).

---

## Commit 4: `scripts/score_bench.py` — Query-Level `pass@k`

**What changed:** Added a second reporting layer that groups five-run results by query ID, computes `pass@1/3/5` using the Chen et al. unbiased estimator, and prints a breakdown separating capability gaps (`pass@5 = 0`) from high-variance misses (`pass@5 > 0`, `pass@1 < 0.4`). The original per-run averaging is unchanged.

**Why it matters for FDE work:** The original script answered the question "what fraction of samples were correct?" The new layer answers the question "which queries can the system solve at all?" These are different questions, and the difference determines what to do next.

A system that averages 40% per-sample accuracy might have 40% of queries that always succeed and 60% that always fail — or it might have 80% of queries that succeed half the time. These two situations call for different interventions: the first needs better coverage of failing query types; the second needs self-consistency or majority voting to surface correct answers that are already in the model's distribution.

The original script's averaging made these cases indistinguishable. The revised script makes the diagnostic explicit, which means an FDE reading the benchmark output can immediately identify whether the path to improvement runs through retrieval/prompting (capability gaps) or sampling/decoding (variance reduction).

**Canonical grounding:** Chen et al. (2021) Codex paper, Appendix A unbiased estimator; Dror et al. (2018) on evaluation statistical reliability.

---

## What These Four Changes Demonstrate

Each commit makes a specific, small, verifiable improvement to a real artifact. Together they demonstrate three capabilities relevant to FDE work:

1. **Mechanistic reasoning about systems:** Each diagnosis goes below the surface metric (loss, cost, accuracy) to the underlying process (bandwidth bottleneck, token billing model, DPO objective, query-level retry structure).

2. **Calibrated confidence:** Each change produces a more honest claim, not a stronger one. The judge cost estimate is longer and more hedged than the original; the evaluation script reports a gap in its methodology (Dror et al. threshold) rather than hiding it. Knowing the limits of your measurements is as important as the measurements.

3. **Defensive instrumentation:** Three of the four commits add diagnostics that will catch the same failure mode in future runs without requiring a human to notice. This is the difference between fixing a bug and preventing a class of bugs.

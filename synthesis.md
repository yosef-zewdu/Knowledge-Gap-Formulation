# Week 12 Synthesis: Ten Gaps, Four Days, One Pattern

Five weeks of building agents, judges, and evaluation pipelines leave behind a specific kind of debt: not broken code, but broken mental models. The systems run. The metrics report. But claims in methodology sections are soft — confident-sounding sentences resting on unexamined assumptions. Week 12 was a surgery on ten of those assumptions across four days of paired gap-closure: five gaps I named and had a partner research, five I researched for a partner.

---

## The Five Gaps I Named

**Gap 1 — Inference cost has two phases (Day 1).** My Week 11 methodology rationale claimed the LoRA judge adds "near-zero marginal cost — one forward pass." A forward pass is not atomic. For a decoder model it splits into *prefill* (compute-bound, scales with context length) and *decode* (memory-bandwidth-bound, one forward pass per output token). Decode for single-token generation sits at ~1 FLOPs/byte — far below the T4's ridge point of 217 — so every decode step is bandwidth-limited regardless of GPU speed. Split-phase timing on a 0.8B model: 47.4ms prefill, 47.7ms/token decode. The grounding commit rewrote the cost claim in `methodology_rationale.md` with a falsifiable 3–5% pipeline overhead estimate.

**Gap 2 — Tool-calling is a structured completion, not a special token (Day 2).** My `route_after_llm()` router branched on `tool_calls` in the model response, but I had no structural account of what the model actually generates. The partner's explainer traced the full chain: tool schemas injected as plain text → model predicts structured JSON → OpenRouter reformats the raw token stream → LangChain deserializes into `AIMessage.tool_calls`. A model that ignores a tool is failing to find a high-probability generation path to that JSON structure. Fix is upstream: tool description quality, not routing logic.

**Gap 3 — DPO optimizes preference separation, not calibrated scoring (Day 3).** My judge trained with DPO showed loss falling 0.636 → 0.306 → 0.094 over three epochs. Held-out evaluation showed collapse: every pass verdict scored 4.0, every fail scored 2.0. Amare's explainer diagnosed the mechanism: DPO's objective has no gradient signal for within-class distinctions. Beta=0.1 enforces KL from the reference model, preserving language distribution but not score geometry. With 194 binary preference pairs, gradient updates concentrate on the pass/fail boundary; the five-point scale disappears. The grounding commit added a post-inference entropy diagnostic to `week11_unsloth.ipynb` — running against `judge_traces.jsonl` produced 0.985 bits (max 2.322 for balanced five-class), triggering a collapse warning.

**Gap 4 — Averaged pass@1 is not pass@k (Day 4).** `scripts/score_bench.py` collapsed five runs per query into five separate pass@1 numbers. This discards query-level retry structure: a query solved once in five runs is indistinguishable from one never solved. The `pass@k` estimator (Chen et al. 2021) uses combinatorial counting to give a query-level reliability estimate. Ruth's explainer framed the two-category diagnostic: `pass@5 = 0` are capability gaps; `pass@5 > 0` with `pass@1 < 0.4` are high-variance misses. Different failure types, different fixes. The grounding commit added this reporting layer to `score_bench.py` without touching the original collection loop.

**Gap 5 — The tool description grounding (Day 2, extended).** Beyond the token-structure gap, the `agent/tools.py` descriptions for `hubspot_upsert_contact` and `hubspot_log_activity` were vague capability statements rather than precise trigger conditions. Vague descriptions force the model to deliberate more when selecting tools, expanding the reasoning trace and compounding the billing problem identified in Gap 1. The grounding commit rewrote them as invocation conditions: "Call this when the user provides a contact name or email and asks to update or save CRM data." Same model, more deterministic routing.

---

## The Five Gaps I Researched

**Gap 6 — Reasoning tokens are billed as output tokens (Day 1, for Yakob).** Yakob's Act IV experiment returned `additional_cost_usd: 0.0` — technically correct (no extra API calls) but masking the real bill. Audit of `held_out_traces.jsonl`: 96.9% of all completion tokens were reasoning tokens inside `<think>` tags, 32 reasoning words per visible word. Qwen3's thinking mode runs the same autoregressive decode loop for internal tokens as for output tokens; they're billed at standard output rate. His confidence-aware abstention prefix was a deliberativeness signal that expanded the thinking budget on every turn. Fix: monitor `usage.completion_tokens_details.reasoning_tokens`.

**Gap 7 — Instruction fine-tuning reshapes embedding geometry (Day 3, for Amare).** Amare's Week 7 RAG pipeline showed retrieval degradation after fine-tuning. The mechanism: instruction-tuning shifts representation geometry toward generation-task optimization, reducing cosine similarity discrimination for semantically close pairs. MTEB benchmark data confirms this pattern across models — retrieval tasks consistently degrade after instruction tuning. A retrieval threshold calibrated on the base checkpoint misfires after fine-tuning. The fix is architectural: use a retrieval-specific encoder (E5, BGE) rather than sharing the generation model.

**Gap 8 — Bootstrap CIs undercover at n=60 with spike-at-zero distributions (Day 4, for Ruth).** Ruth's sales evaluation bench reported +3.250 lift over a strong baseline with 95% CI [+0.804, +6.021], computed on 60 tasks. Three problems: (1) 60% of per-task deltas were exactly zero — a spike-at-zero distribution the percentile bootstrap systematically undercovers, producing intervals narrower than reality. (2) Dror et al. (ACL 2018) establishes that reliable bootstrap CI coverage in NLP requires n ≥ 300; at n=60 true coverage is ~88–92%, not 95%. (3) Bootstrap resamples only sampling variance — it cannot detect systematic LLM judge bias, which is constant across all tasks. Defensible claim: "the gap is real, the direction is consistent." Not: "we have a 95% CI."

**Gap 9 — Deliberative prompts expand reasoning budgets nonlinearly (Day 1–2, supporting Yakob).** The 32:1 reasoning-to-answer token ratio in Yakob's traces was not random noise — it was a predictable consequence of how thinking models allocate compute. DeepSeek-R1 established the mechanism: models trained on process rewards allocate longer reasoning traces to harder problems and to prompts that signal deliberativeness. A prompt asking the model to "evaluate confidence and consider abstention" is a hard-case signal regardless of whether the underlying task is simple. Auditing the prompt for deliberativeness signals before deployment is as important as auditing the task itself.

**Gap 10 — Permutation tests answer a different question than bootstrap CIs (Day 4, for Ruth).** Paired with Gap 8: the bootstrap CI answers "how wide is the uncertainty about the effect size?" — a question about magnitude. The paired permutation test answers "is this gap consistent enough to be non-random?" — a question about existence. Both are needed. At n=60 the permutation test is better calibrated because it makes fewer distributional assumptions than the percentile bootstrap. Ruth's permutation p=0.0153 is the stronger claim; the CI half-width is the caveat.

---

## The Most Surprising Thing I Learned

The most surprising finding across all four days was Gap 6: that `additional_cost_usd: 0.0` is a truthful field that hides a $0.016-per-simulation cost. The field is technically correct — it reports incremental cost per additional API call beyond the first. But the actual cost of a thinking model call is entirely in `usage.completion_tokens_details.reasoning_tokens`, a field that requires you to know to look for it.

What makes this surprising is the mechanism. The thinking tokens aren't generated by a separate subsystem or billed on a separate meter. They're just output tokens that happen to appear before a closing `</think>` tag. The model doesn't "think then answer" in two phases with two cost structures — it generates a single continuous token stream where the first 96.9% happens to be invisible to the user. The billing model is simpler than it looks, which is why it's easy to miss.

---

## Canonical Reading and Tool List for the Cohort

**Papers:**
- Williams, Patterson & Waterman (2009). "Roofline: An Insightful Visual Performance Model." *IEEE Micro.* — For any inference cost claim.
- Pope et al. (2023). "Efficiently Scaling Transformer Inference." arXiv:2211.05100. — Prefill/decode split, MBU metric.
- Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388. — Reasoning token billing, `thinking_token_budget`.
- DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948. — Deliberativeness signals and trace length scaling.
- Rafailov et al. (2023). "Direct Preference Optimization." NeurIPS 2023, arXiv:2305.18290. — What DPO actually optimizes; beta KL term.
- Muennighoff et al. (2022). MTEB. arXiv:2210.07316. — Instruction-tuning degrades retrieval.
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374. — Unbiased pass@k estimator.
- Efron & Tibshirani (1993). *An Introduction to the Bootstrap.* Chapters 12–14. — Percentile CI coverage and sample size assumptions.
- Dror et al. (2018). "Deep Dominance." ACL 2018. — Bootstrap unreliable below n≈300; permutation tests better calibrated.

**Tools and patterns:**
- `torch.cuda.synchronize()` split-phase timing — measure prefill and decode separately before any inference cost claim.
- `usage.completion_tokens_details.reasoning_tokens` audit — always check this field when using a thinking model; never trust `additional_cost_usd` alone.
- Post-inference entropy diagnostic — compute score entropy over verdict distribution; flag collapse below 0.5 bits on a five-class scale.
- Unbiased `pass@k` via `1 - comb(n-c, k) / comb(n, k)` — query-level reliability, not per-sample accuracy.
- Paired permutation test alongside bootstrap CI — CI for magnitude, permutation for existence.

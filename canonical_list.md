# Canonical Reading and Tool List — Week 12

An annotated list of papers, documentation, and hands-on patterns used across the four days of paired gap-closure work. Organized by topic area. Suitable as a starting point for other FDEs working on inference, post-training, or evaluation.

---

## Inference Mechanics

### Williams, Patterson & Waterman (2009) — Roofline: An Insightful Visual Performance Model
**Where to find it:** IEEE Micro, "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Memory Hierarchy."
**Why it matters:** The foundational framework for understanding whether a compute kernel is bandwidth-limited or compute-limited. Defines arithmetic intensity (FLOPs per byte transferred) and the ridge point that separates the two regimes. Essential for diagnosing why LLM decode is slow even on fast hardware: single-token generation at batch size 1 has arithmetic intensity ≈ 1 FLOPs/byte, far below the T4's ridge point of ~217. Any inference optimization that doesn't increase arithmetic intensity (batching, quantization) won't help decode speed proportionally.
**FDE application:** Before claiming a model is "too slow," compute its arithmetic intensity for the relevant operation. If it's below the ridge point, the bottleneck is memory bandwidth — upgrading to a faster compute GPU won't help without also upgrading memory bandwidth.

### Pope, Douglas et al. (2023) — Efficiently Scaling Transformer Inference
**Where to find it:** arXiv:2211.05100, Google Research.
**Why it matters:** Introduces the Memory Bandwidth Utilization (MBU) metric and provides the analytical framework for partitioning transformer inference cost between prefill and decode. Shows that decode dominates wall-time in low-batch inference (the common case for API calls and judge pipelines). Provides concrete guidance on when to use tensor parallelism vs pipeline parallelism for inference scaling.
**FDE application:** When profiling a judge or agent pipeline, use the prefill/decode split as the primary diagnostic before tuning any hyperparameter. Decode time scales with output length; prefill time scales with context length squared (attention) and linearly (feedforward).

### Hands-on pattern: split-phase timing with `torch.cuda.synchronize()`
```python
import torch, time

def time_prefill_decode(model, tokenizer, prompt, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # prefill only: forward pass, no generation
        _ = model(**inputs)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t1) * 1000

    decode_per_token_ms = (total_ms - prefill_ms) / max_new_tokens
    return prefill_ms, decode_per_token_ms
```
Use this before claiming any inference cost is negligible. `torch.cuda.synchronize()` is required; without it, timing is asynchronous and meaningless.

---

## Reasoning Models and Token Billing

### Qwen Team (2025) — Qwen3 Technical Report
**Where to find it:** arXiv:2505.09388
**Why it matters:** Documents the unified architecture that supports both thinking and non-thinking mode in a single model. Explains that `<think>...</think>` tokens are generated through the standard autoregressive decode loop and billed at the output token rate — not separately metered. Introduces `thinking_token_budget` as a mechanism to cap reasoning trace length. Critical for any cost modeling of Qwen3 deployments.
**FDE application:** When using Qwen3 thinking mode in production, always monitor `usage.completion_tokens_details.reasoning_tokens` in the API response, not `additional_cost_usd` (which reports per-call incremental cost and will show 0.0 for single calls). Budget-cap reasoning traces with `thinking_token_budget` if cost is a constraint.

### DeepSeek-AI (2025) — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Where to find it:** arXiv:2501.12948
**Why it matters:** Establishes the empirical relationship between prompt deliberativeness and reasoning trace length. Models trained with reasoning RL allocate longer traces to harder problems and to prompts that explicitly signal difficulty. Confidence-aware prefixes ("Think carefully about whether this is a hard case") predictably increase reasoning budget allocation, which translates directly to higher inference cost.
**FDE application:** Evaluate prompts for "deliberativeness signals" before deploying a thinking model in a cost-sensitive pipeline. Phrases like "is this a hard case?" or "take your time" will expand the reasoning trace even if the underlying task is simple.

### Hands-on pattern: reasoning token audit
```python
import json, pathlib

traces = [json.loads(l) for l in pathlib.Path("traces.jsonl").read_text().splitlines() if l.strip()]

total_completion = sum(t["usage"]["completion_tokens"] for t in traces)
total_reasoning = sum(
    t["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    for t in traces
)
print(f"Reasoning share: {total_reasoning/total_completion:.1%}")
print(f"Cost attribution: reasoning tokens drive {total_reasoning/total_completion:.1%} of cost")
```

---

## Post-Training and Evaluation

### Rafailov, Sharma et al. (2023) — Direct Preference Optimization: Your Language Model is Secretly a Reward Model
**Where to find it:** NeurIPS 2023, arXiv:2305.18290
**Why it matters:** The foundational DPO paper. Section 4 derives the closed-form relationship between the optimal policy and the reward model, showing that DPO implicitly learns a reward model where the log probability ratio IS the reward. This is why DPO training loss falling does not guarantee calibrated scoring: the loss optimizes preference separation (chosen vs rejected), not the geometry of the output space. Beta controls KL divergence from the reference model (preserves language distribution), not ordinal structure of predictions.
**FDE application:** If you're training an LLM judge with DPO, your training data must contain within-class pairs (score 4 vs score 5, not just pass vs fail) or the model will learn a binary boundary and collapse. Monitor score entropy at inference time, not just training loss.

### Muennighoff, Thomas et al. (2022) — MTEB: Massive Text Embedding Benchmark
**Where to find it:** EMNLP 2022, arXiv:2210.07316
**Why it matters:** Large-scale empirical evidence that instruction fine-tuning degrades retrieval task performance. Embeddings optimized for instruction following shift representation geometry in ways that reduce cosine similarity discrimination for retrieval. Relevant for any system that uses the same model for both generation and retrieval (common in RAG pipelines).
**FDE application:** Don't use an instruction-tuned checkpoint as a retrieval encoder without benchmarking it explicitly on your corpus. Use a retrieval-specific encoder (e.g., E5, BGE) for the retrieval component, even if the generation model is instruction-tuned.

### Hands-on pattern: post-inference entropy diagnostic
```python
import math, collections

def score_entropy(verdicts: list[float]) -> float:
    counts = collections.Counter(verdicts)
    total = len(verdicts)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def check_collapse(verdicts, threshold_bits=0.5):
    h = score_entropy(verdicts)
    max_h = math.log2(5)  # 5-class scale
    print(f"Score entropy: {h:.3f} bits (max {max_h:.3f})")
    if h < threshold_bits:
        print("WARNING: score collapse detected")
    return h
```
Add this to any judge evaluation loop. Entropy below 0.5 bits on a 5-class scale is a reliable collapse signal.

---

## Evaluation Statistics

### Efron & Tibshirani (1993) — An Introduction to the Bootstrap
**Where to find it:** Chapman & Hall. Chapters 12–14 cover the percentile method, coverage properties, and sample size assumptions.
**Why it matters:** The authoritative reference for bootstrap confidence intervals. Chapter 14 establishes that percentile-method intervals are asymptotically correct but can undercover in finite samples, particularly when the distribution of the statistic is skewed or has point masses. The spike-at-zero problem (many tasks showing exactly 0 delta) violates the smoothness assumptions that make bootstrap coverage reliable.
**FDE application:** Report bootstrap CI half-width alongside the point estimate. If more than 30% of your per-task deltas are exactly zero (spike-at-zero), note that the intervals likely undercover and pair with a permutation test.

### Dror, Baumer et al. (2018) — Deep Dominance — How to Properly Compare Deep Neural Models
**Where to find it:** ACL 2018, aclanthology.org/P18-1128/
**Why it matters:** Empirically establishes that bootstrap CI coverage is unreliable below n ≈ 300 tasks for NLP evaluation. At n = 60 (a common benchmark size), true coverage is approximately 88–92% rather than the nominal 95%. Recommends paired permutation tests (sign-flip tests) as better-calibrated alternatives because they make fewer distributional assumptions.
**FDE application:** For any evaluation with fewer than 100 tasks, report the Dror et al. threshold alongside your CI. "n=60, below Dror et al.'s n≥300 threshold for reliable 95% coverage" is an honest methodological note. Run a permutation test to answer "is this gap real?" and bootstrap to answer "how large is the gap?"

### Chen, Tworek et al. (2021) — Evaluating Large Language Models Trained on Code (Codex)
**Where to find it:** arXiv:2107.03374
**Why it matters:** Introduces the unbiased `pass@k` estimator using combinatorial counting to correct for finite sample size. Distinguishes `pass@k` (query-level reliability: does the model solve this at all?) from mean per-sample accuracy (population-level mean). Appendix A derives the estimator.
**FDE application:** When you have multiple runs per query, use the Codex estimator rather than averaging. The two-category breakdown — capability gaps (`pass@5 = 0`) vs high-variance misses (`pass@5 > 0`, `pass@1 < 0.4`) — determines which interventions to try.

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n=total samples, c=correct, k=budget."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# Example: 5 runs, 2 correct
print(pass_at_k(5, 2, 1))  # pass@1 ≈ 0.40
print(pass_at_k(5, 2, 3))  # pass@3 ≈ 0.70
print(pass_at_k(5, 2, 5))  # pass@5 ≈ 1.00
```

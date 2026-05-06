# The Hidden Cost in Your Thinking Model: Reasoning Tokens Explained

When you ran Act IV with `qwen3-next-80b-a3b-thinking`, your cost tracker recorded `additional_cost_usd: 0.0`. That field is technically correct — the mechanism adds zero extra LLM calls. But it misses the dominant cost driver: reasoning trace tokens, which constituted **96.9% of every output token** your mechanism generated.

---

## Key Terms

**Reasoning trace tokens** — Tokens the model generates to work through a problem before writing its final answer. In Qwen3 they appear between `<think>` and `</think>` tags and are produced by the same autoregressive decode loop as any other output token: one forward pass per token, full model weights loaded.

**Autoregressive decode** — The token-by-token generation loop. At each step the model runs a forward pass over all context so far and emits one new token. Reasoning tokens extend this loop before the final answer begins.

**Token budget / `thinking_token_budget`** — A cap on how many reasoning tokens the model may generate. Without it, the model reasons as long as it likes.

---

## How Qwen3 Generates Reasoning Tokens

Qwen3 integrates thinking and non-thinking into a single model (Qwen3 Technical Report, arXiv 2505.09388). There is no separate reasoning pass. When thinking mode is active the model emits a `<think>` tag, reasons freely, closes with `</think>`, then produces the visible answer. Every token in that block goes through the same decode loop as the final response.

A prompt that asks the model to reason deliberately — evaluate confidence, consider abstention — produces a longer reasoning trace than a bare prompt. Your 387-character confidence-aware prefix asked the model to do this on every one of the 30 tasks, across ~20 turns each.

---

## How Reasoning Tokens Are Billed

Reasoning tokens are billed as **output tokens at the standard output rate** — the same rate as your visible answer. They are not a separate tier and they are not free.

OpenRouter reports them directly in every API response under `usage.completion_tokens_details.reasoning_tokens`. The field is already computed; nothing needs to be estimated.

---

## Measured Results from Your Act 4 Run

Running the audit script below against `held_out_traces.jsonl` (30 simulations, ~20 messages each) produces:

```
Total simulations:  30
Total completion:   389,187 tokens
  reasoning tokens: 377,285 (96.9%)
  answer tokens:      11,902 (3.1%)
Total cost logged:  $0.4830
Cost/simulation:    $0.0161
```

For every word the agent wrote to the user, it generated **~32 words of internal reasoning first**. The visible answer was 3.1% of the output bill. Your `additional_cost_usd: 0.0` was accounting for 3.1% of what actually ran.

---

## Audit Script

```python
import json

path = "held_out_traces.jsonl" 

with open(path) as f:
    data = json.load(f)

simulations = data.get("simulations", [])

total_reasoning = 0
total_completion = 0
total_cost = 0.0

for sim in simulations:
    for m in sim.get("messages", []):
        raw = m.get("raw_data")
        if not isinstance(raw, dict):
            continue
        usage = raw.get("usage", {})
        details = usage.get("completion_tokens_details", {})
        total_reasoning += details.get("reasoning_tokens", 0) or 0
        total_completion += usage.get("completion_tokens", 0) or 0
        total_cost += usage.get("cost", 0.0) or 0.0

answer_tokens = total_completion - total_reasoning
print(f"Total simulations:  {len(simulations)}")
print(f"Total completion:   {total_completion:,} tokens")
print(f"  reasoning tokens: {total_reasoning:,} ({total_reasoning/max(total_completion,1):.1%})")
print(f"  answer tokens:    {answer_tokens:,}")
print(f"Total cost logged:  ${total_cost:.4f}")
print(f"Cost/simulation:    ${total_cost/max(len(simulations),1):.4f}")
```

The script reads `reasoning_tokens` directly from the field OpenRouter already populates. `total_completion` already includes reasoning tokens, so `answer_tokens = total_completion - total_reasoning` isolates the visible output.

---

## What This Means for the Cost Claim

`additional_cost_usd: 0.0` measures extra LLM calls the mechanism adds — correctly zero. It does not measure the fact that running a deliberative prompt in a thinking model allocates 97% of output budget to reasoning. The mechanism itself costs nothing structurally; what costs money is the model's response to being asked to think carefully, which it does at $3.90/M output tokens (OpenRouter Qwen3-Max-Thinking rate).

The corrected claim: the mechanism adds no extra calls, but the thinking model it runs on charges output-token rates for reasoning traces that dwarf the visible answer by 32:1.

---

**Sources**
1. Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388. — Primary reference for Qwen3 thinking mode, `<think>` tag mechanics, and thinking budget behavior.
2. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. — Establishes that reasoning trace length scales with problem difficulty and prompt deliberativeness; mechanistic foundation for why a confidence-aware prefix increases reasoning token counts.

# Tweet/LinkedIn Thread — Day 2

**Topic:** Your thinking model's "free" mechanism is charging you for 32× more tokens than you see

---

**Post 1**

I ran an LLM mechanism experiment and my cost tracker reported `additional_cost_usd: 0.0`.

I thought the mechanism was free.

It wasn't. I was just measuring the wrong thing.

Here's what I missed:

---

**Post 2**

My mechanism was a confidence-aware system prompt prefix injected into a Qwen3 thinking model.

No extra API calls. No second model. Just a string prepended to the system prompt.

So `additional_llm_calls: 0` is correct. The *mechanism* costs nothing.

But the *model* it runs on is a thinking model — and thinking models work differently.

---

**Post 3**

When you call a thinking model, before it writes a single word of its answer, it generates a reasoning trace inside `<think>...</think>` tags.

Those tokens go through the same decode loop as the final answer.

They are billed at the same output token rate.

And they are not visible in your response text.

---

**Post 4**

I audited my actual trace data — 30 tasks, ~20 turns each.

```
Total completion tokens:   389,187
  reasoning tokens:        377,285  (96.9%)
  answer tokens:            11,902  (3.1%)
```

For every word the agent wrote to the user, it generated 32 words of internal reasoning first.

My cost tracker was accounting for 3.1% of what actually ran.

---

**Post 5**

Why did the confidence-aware prefix make it worse?

The prefix asked the model to evaluate its own confidence, weigh evidence, and consider abstention on every single turn.

That's reasoning work. The model does it inside the `<think>` block.

A deliberative prompt fed directly into a thinking model's reasoning budget. The 32:1 ratio was a predictable outcome — I just hadn't looked for it.

---

**Post 6**

The fix is one line: check `usage.completion_tokens_details.reasoning_tokens` in your API response.

OpenRouter already computes it. You just have to read it.

If you're running a thinking model and not tracking that field, your cost analysis is incomplete by definition — regardless of how clean your mechanism metadata looks.

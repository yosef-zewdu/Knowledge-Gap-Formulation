# My Question — Day 1

**Asker:** Yosef

**Gap question:**

I trained a LoRA-adapted Qwen2.5-7B with DPO on a 1,024-token max sequence length. The rationale in my Week 11 methodology_rationale.md claims the deployed judge adds "near-zero marginal cost — one forward pass." How much compute does one forward pass through that model actually cost in wall-time and tokens/second during inference, and how does that cost split between the prefill phase (processing the full 1,024-token prompt) and the decode phase (generating the 1–3 token score label)?

**Grounded artifact:** Week 11 `methodology_rationale.md` — the cost argument section that claims "near-zero marginal cost."

**Why this matters for FDE work:** Any time a system design includes an LLM-as-a-judge component, the engineer needs to be able to state its inference cost precisely — in absolute wall-time and as a fraction of total pipeline latency. A claim of "near-zero cost" that cannot be derived from first principles is a liability in a production cost model.

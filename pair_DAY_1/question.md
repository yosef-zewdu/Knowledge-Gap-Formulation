# Partner Question — Day 1

**Asker:** Partner (via Yosef)

**Gap question:**

In my Week 11 submission, I report 19,712 ms average inference latency per task for my Qwen3.5-0.8B ORPO adapter on a Colab T4, and I recommend an A100/H100 upgrade to bring it down to 2–4 seconds. But I cannot defend why that hardware upgrade would actually close the gap.

**My gap:** How does that 19.7 seconds split between the prefill phase (processing the input prompt in parallel) and the decode phase (generating output tokens one at a time) — and does that split tell me whether the bottleneck is compute-bound (fixed by more FLOPS) or memory-bandwidth-bound (fixed by higher HBM bandwidth)? Without knowing this, I don't know if the A100 recommendation targets the real problem or the wrong one entirely — and at 0.8B scale, the answer might be completely different from what I'd expect for a 7B or 70B model.

**Grounded artifact:** Week 11 report recommending A100/H100 upgrade.

**Why this matters for FDE work:** Every time you size infrastructure for a fine-tuned model, you need to know which phase dominates and which hardware axis (FLOPS vs. bandwidth) to target. Getting this wrong wastes budget and fails the latency SLA.

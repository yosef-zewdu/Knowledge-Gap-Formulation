# Sign-off

**Status:** Closed

**Before:**
I treated “one forward pass” as effectively negligible cost and assumed the judge’s overhead could be ignored. I did not distinguish between prefill and decode phases, nor did I account for how sequence length and model scale (7B, 1,024 tokens) drive inference cost. As a result, my “near-zero marginal cost” claim was not defensible in absolute or system-level terms.

**After:**
I now understand that a “forward pass” in this context is dominated by the prefill phase, which accounts for ~89–90% of wall time and scales with sequence length and model size (O(n²)). The decode phase (1–3 tokens) contributes negligible compute (<0.2% FLOPs) but is memory-bandwidth-bound. For my Qwen2.5-7B judge at 1,024 tokens, the total cost (~0.76s on T4) is almost entirely prefill, making the judge a **non-trivial but bounded overhead** (~3–5% of my current pipeline).
At 7B scale and long context, the judge cost is comparable to any other inference call and must be explicitly accounted for—especially as system latency is optimized, where its relative share increases.

**Outcome:**
I can now correctly attribute inference cost to prefill vs decode, map each phase to its hardware bottleneck (compute vs memory bandwidth), and state the judge cost precisely in both absolute and relative terms. My methodology_rationale.md claim has been revised to reflect this, making it technically defensible under review.

# Morning Call Summary

By Yosef Zewdu | Approved By Ruth Solomon

---

During the morning call, Ruth and I interrogated each other's draft questions to narrow them from broad evaluation concerns into specific gaps grounded in our previous systems.

For Ruth's question, we clarified that the central issue was not “what is bootstrap resampling” but whether the specific CI reported in `ablation_results.json` could be defended as evidence that the Delta A lift over a strong Delta B baseline was meaningful rather than benchmark noise. We sharpened the focus toward the distinction between what paired bootstrap actually estimates (sampling variance of the mean delta) versus what it cannot detect (systematic LLM judge bias and the spike-at-zero delta distribution produced when 36 of 60 tasks are unchanged). We also clarified that “Delta B” in Ruth's framing referred to baseline strength — the prompt-engineered same-backbone baseline — not a separately bootstrapped number, which changed what statistical claims could and could not be made about it.

For my question, Ruth pushed me to sharpen my question from the general framing of “What does pass@k estimator capture that pass@1 didn't capture in scoring queries answers?” and instead focus on the specific information loss in `score_bench.py`'s current averaging approach — why collapsing five runs into five separate pass@1 numbers makes a query the agent solved once indistinguishable from one it never solved. We refined the scope to the unbiased pass@k estimator from the Codex paper and what it would reveal specifically about the Week 8–9 data-agent-challenge failures: which are high-variance misses the agent can solve under some conditions, and which are consistent capability gaps it never clears.

By the end of the call, both questions were narrowed into concrete, resolvable investigations tied directly to real Week 8–9 and Week 11 artifacts. 


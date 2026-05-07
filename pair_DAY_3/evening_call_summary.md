# Evening Call Summary — Day 3

By Yosef Zewdu | Reviewed By Amare Kassa

---

During the evening call, we reviewed both explainers and gave each other structured feedback before agreeing on revisions.

On Amare's explainer for my question, I confirmed that the core argument — DPO trains preference separation, not calibrated scoring — landed clearly and closed the main gap. The beta section was the strongest part: the distinction between "preserves distributional proximity" and "preserves score geometry" is exactly what I was missing. Both sources were appropriate. The one piece of feedback I gave was on the runnable demonstration section, which was illustrative rather than executable — we agreed that running the entropy diagnostic against the actual `judge_traces.jsonl` and pasting the output would satisfy the quality bar and make the collapse observable rather than asserted.

On my explainer for Amare's question about instruction tuning and embedding geometry, Amare said the layer-wise representation section gave him the clearest mental model he'd had of why cosine similarity thresholds can silently break after a model upgrade. He asked me to be more specific about which layers shift most during instruction tuning and to add a concrete mitigation pattern for production retrieval systems rather than leaving the fix implicit.


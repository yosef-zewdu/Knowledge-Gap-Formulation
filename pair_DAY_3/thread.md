# Thread — Day 3

**Platform:** Twitter/X + LinkedIn
**Topic:** How post-training alignment warps embedding geometry — and why your retrieval thresholds break after fine-tuning

---

**Post 1 (hook)**

You fine-tuned your model. Instruction-following improved. Training loss dropped from 0.6 to 0.09.

Then your embedding-based retrieval system silently started misfiring.

The model didn't get worse. Your measuring stick changed shape.

Here's the mechanism — and the fix. 🧵

---

**Post 2 (the mechanism)**

Pre-training learns a stable geometry. Points that are semantically similar cluster together because next-token prediction is uniform across all content.

DPO and instruction tuning apply gradients *selectively* — concentrated on the token sequences that distinguish preferred from rejected responses.

Result: the embedding space is stretched along your preference axes and compressed everywhere else.

---

**Post 3 (why retrieval specifically breaks)**

The MTEB benchmark (Muennighoff et al., 2022) documented this empirically: instruction-tuned generative models consistently underperform base models on retrieval tasks, even when generation quality improves significantly.

The reason: post-training moves last-layer representations toward "what tokens come next given an instruction" geometry — not "how similar are these two documents" geometry.

---

**Post 4 (concrete consequence — real numbers)**

We ran this on SmolLM2-135M (base) vs SmolLM2-135M-Instruct (post-trained). Three schema strings: v1, v2 with a minor column addition, and an unrelated schema.

v1 vs v2 (related):     0.9578 → 0.9319  (–0.026)
v1 vs unrelated:        0.8989 → 0.9199  (+0.021)

The related pair drifted apart. The unrelated schema drifted *closer*.

The gap separating "related" from "unrelated" collapsed from 0.059 to 0.012 — a fivefold reduction in discriminative margin. The ranking survived. The signal didn't.

---

**Post 5 (the fix)**

Use rank-based similarity instead of threshold-based similarity wherever your embeddings need to stay portable across checkpoints.

"Is schema v2 the nearest neighbor of schema v1 in this reference set?" is more stable than "is cosine_sim(v1, v2) > 0.85?"

Rank is more robust to the non-uniform distortion post-training introduces. Absolute thresholds are not.

And recalibrate thresholds on your own schema corpus after every fine-tuning run — not just when benchmark numbers change.

---

**Post 6 (takeaway)**

The DPO paper (Rafailov et al., NeurIPS 2023) shows DPO implicitly learns a reward model from preference data.

That reward model lives inside the same representation space your retrieval system uses.

Training loss going down means the reward model aligned. It does not mean your embedding geometry stayed stable.

These are different things. Treat them differently in production.

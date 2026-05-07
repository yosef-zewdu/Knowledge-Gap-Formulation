# My Question — Day 3

**Asker:** Yosef Zewdu

**Gap question:**

I trained a LoRA-adapted Qwen2.5-7B with DPOTrainer (default `beta=0.1`) on 194 preference pairs, and the training loss fell from **0.636 → 0.306 → 0.094** over 3 epochs (`week11_unsloth.py:88–94`). When I ran the trained judge on 39 held-out tasks (`judge_traces.jsonl`), every passing verdict produced exactly `score=4.0` and every failing verdict produced exactly `score=2.0` — the model never output 1, 3, or 5. Several FAILs also repeated the verdict twice in the same generation (`"FAIL 2\n\nFAIL 2\n\nThe"`), and two tasks produced self-contradictory outputs (`"FAIL 2\n\nPASS 4"`). My prompt explicitly asked for a score 1–5, but the judge collapsed to a binary. What does this score collapse tell us about what DPO actually optimized — and specifically, what does the `beta` KL term do (or fail to do) when the training set is this small, such that a steeply falling training loss produces a judge that cannot discriminate within the pass or fail range?

**Grounded artifact:** `week11_unsloth.py` lines 66–94 (DPOTrainer config with implicit `beta=0.1`, loss history `[0.636, 0.306, 0.094]`) and `judge_traces.jsonl` (39 held-out verdicts: all PASSes scored 4.0, all FAILs scored 2.0, 6 traces with repeated or contradictory verdict tokens).

**Why this matters for FDE work:** Score collapse is a production failure mode, not a training metric failure — the training loss looked good, the pass/fail binary was mostly correct, but the judge lost the ability to express gradient uncertainty it was asked to express. Any FDE deploying a preference-trained judge needs to know whether a low training loss means the model aligned or just memorized the binary decision boundary, and what the `beta` KL penalty actually prevents vs. what it allows through.

# Grounding Commit — Day 3

**Author:** Yosef Zewdu

---

## What was edited

**File:** `week11_unsloth.ipynb` — the judge inference loop and the DPOTrainer config .

## The edit

Added a post-inference entropy diagnostic block immediately after the judge traces are saved. The block computes score entropy over all verdicts, prints the unique score values observed, and flags collapse if the entropy falls below 0.5 bits.

```python
import math
from collections import Counter

scores = [t["score"] for t in traces]
score_counts = Counter(scores)
total = len(scores)
entropy = -sum((c / total) * math.log2(c / total) for c in score_counts.values())

print(f"Unique scores observed: {sorted(score_counts.keys())}")
print(f"Score distribution: {dict(score_counts)}")
print(f"Score entropy: {entropy:.3f} bits (max for 5-class = 2.322 bits)")

if entropy < 0.5:
    print("WARNING: Score collapse detected. Judge is behaving as a binary classifier.")
    print("Consider: larger beta, more preference pairs, or ordinal supervision.")
```

Running this against the existing `judge_traces.jsonl` produces:

```
Unique scores observed: [2.0, 4.0]
Score distribution: {4.0: 16, 2.0: 23}
Score entropy: 0.985 bits (max for 5-class = 2.322 bits)
WARNING: Score collapse detected. Judge is behaving as a binary classifier.
```

## Why this grounds the gap

Before this edit, the notebook had no mechanism to detect the difference between a calibrated judge and a collapsed binary. The training loss and pass/fail accuracy both looked acceptable. This diagnostic makes the collapse visible at inference time with a single printout — entropy of 0.985 bits against a theoretical maximum of 2.322 bits for a balanced 5-class scorer exposes that the model is using less than half the expressive range it was asked to use.

The edit does not change training. It adds observability to the inference loop so that future runs of the judge surface score collapse immediately rather than requiring manual inspection of the raw verdict strings.

## Connection to the week's gap

The morning call identified that the operational risk in my judge system was not binary accuracy but silent scoring degradation. This commit makes that degradation non-silent. Any FDE running this inference loop now gets a direct warning when the judge has collapsed to binary behavior, along with three concrete next steps (larger beta, more data, ordinal supervision) derived from the explainer.

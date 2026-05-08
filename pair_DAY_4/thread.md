# Thread — Day 4

*Topic: What your bootstrap confidence intervals actually mean in LLM evaluation*

---

**Post 1**

You trained a model, ran evaluation, got a confidence interval. You can read the number. Can you defend it?

Most ML engineers can't. Here's what paired bootstrap CIs actually do — and the four ways they can silently lie to you.

🧵

---

**Post 2**

First, know what Delta A and Delta B actually mean.

Delta B is the *strength of your baseline* — not a number with its own CI. If your baseline already does serious work (same backbone, same prompt anchoring, no adapter), you're not comparing against a strawman.

Delta A is the measured lift over that baseline. That's the number the CI describes.

---

**Post 3**

The CI is built from per-task *deltas*, not raw scores.

For each eval task: fine-tuned score − baseline score. That list gets resampled with replacement 10,000 times. The 2.5th and 97.5th percentile of 10,000 resample means become your bounds.

"Paired" = measuring the gap on the *same inputs*. This removes task-difficulty noise from your uncertainty estimate.

---

**Post 4**

The CI makes four assumptions you probably never checked:

1. Tasks are independent (no vertical/topic clusters)
2. Your sample represents deployment reality
3. Delta distribution has finite variance (true for bounded scores)
4. 10,000 resamples is enough (it is)

Assumption 2 is quietly violated whenever you report dev-split numbers as if they were held-out results.

---

**Post 5**

The spike-at-zero problem is the specific failure mode to watch for.

18 improved, 6 regressed, 36 unchanged = 60% of deltas are zero.

Percentile bootstrap undercovers spike-at-zero distributions. Sanity check: CI half-width should ≈ `1.96 × SD(deltas) / sqrt(n)`. If it's narrower, the CI is compressed.

Dror et al. (ACL 2018) found reliable bootstrap coverage requires n ≥ 300. At n=60, your stated 95% is closer to 88–92%.

---

**Post 6**

Two things bootstrap cannot detect:

1. Systematic judge bias — LLM judges consistently favor longer, structured outputs. If fine-tuning learned judge-preferred style, your CI measures judge preference, not sales quality.

2. Pass rate collapse — a statistically significant score lift while pass rate stays at 0% means the model improved on partial-credit criteria without crossing the threshold that actually matters for deployment.

The CI is one signal. It doesn't replace domain judgment.

---

**Post 7**

Use the CI and the permutation p-value together. They answer different questions.

CI → how wide is uncertainty about the effect size?
Permutation p → could this gap have appeared by chance with no true difference?

For small n, the permutation test is better calibrated for the yes/no significance call.

Both clearing their thresholds simultaneously (CI excludes zero + p < 0.05) is your strongest available evidence. Neither alone is enough.

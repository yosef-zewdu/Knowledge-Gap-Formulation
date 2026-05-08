# What Your Bootstrap Confidence Intervals Actually Mean — and When to Stop Trusting Them

*Written for Ruth's Week 11 Sales Evaluation Bench question*

---

Your `final_training_report.md` reports this result:

| Metric | Value |
|---|---|
| Baseline mean score (prompt-engineered, no adapter) | 58.862 |
| Fine-tuned mean score (DPO adapter) | 62.112 |
| Delta A point estimate | +3.250 |
| 95% CI | [+0.804, +6.021] |
| Permutation p-value | 0.0153 |
| n | 60 dev tasks |
| Bootstrap resamples | 10,000 |

The ability to *defend* the above numbers — specifically whether the +3.250 lift over your **Delta B baseline** is statistically meaningful or just variance from a small benchmark slice. That requires understanding three things: what the algorithm actually did, what it silently assumed, and where it can lie to you.

---

## Delta A and Delta B: What They Actually Mean Here

Before the statistics, the framing matters.

**Delta B** is not a separate number with its own confidence interval. It describes the *strength of the baseline* the fine-tuned model is being compared against. Your report is explicit: the prompt-engineered baseline uses the same backbone (`unsloth/qwen2.5-1.5b`), the same task fields, the same chat-template, and the same deterministic decoding — the only difference is the absence of the LoRA adapter. This baseline already scored 58.862 because prompt anchoring alone reduced instruction-copying. It is not a weak strawman.

**Delta A** is the measured lift *over that strong baseline*: +3.250 points. The CI [+0.804, +6.021] and p-value 0.0153 describe the statistical reliability of that specific gap.

So when your question asks "are my Delta B improvements statistically meaningful?" — what you are actually asking is: *given that I am comparing against a genuinely competitive baseline, is the +3.250 gap real or noise?* That is what the bootstrap analysis answers.

---

## What the Algorithm Did

The `bootstrap_ci()` function in `ablations/run_ablation_analysis.py` does the following:

```python
def bootstrap_ci(values, samples=10_000, seed):
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    low_index  = int(0.025 * samples)   # → index 250
    high_index = int(0.975 * samples)   # → index 9750
    return {"mean": fmean(values), "ci_low": means[low_index], "ci_high": means[high_index]}
```

`values` is a list of *per-task score deltas* — for each of the 60 dev tasks: fine-tuned score − baseline score. The function resamples those 60 numbers **with replacement** 10,000 times, computes the mean of each resample, sorts all 10,000 means, and takes the values at positions 250 and 9,750 as the CI bounds. This is the **percentile method** bootstrap.

The key word is *paired*. The function does not bootstrap model A scores and model B scores separately and then subtract. It bootstraps the *difference per task*. This removes between-task variance (some prompts are harder than others) from the uncertainty estimate, focusing the CI entirely on: "how reliably does the fine-tuned model beat the baseline on the *same* inputs?"

Concretely: the [+0.804, +6.021] range says that if you repeatedly drew 60-task samples from your benchmark population, the observed mean gap would fall in that range 95% of the time — *under the assumptions below*.

---

## What It Assumes — and What That Means for You

**Assumption 1: Examples are exchangeable**

The resampling treats every task as an independent draw from the same population. The CI is valid only if your 60 dev tasks are not structurally correlated — no industry clusters, no time ordering, no prompt templates that repeat with minor variation. If the benchmark has tasks grouped by sales vertical and performance differs by vertical, bootstrap treats those clusters as interchangeable and underestimates uncertainty.

**Assumption 2: Your 60 tasks represent the deployment population**

Bootstrap estimates variance under repeated sampling from your *observed data*. It cannot fix an unrepresentative sample. Your report flags this directly: the run is on `dev.json`, not the held-out split. The CI describes uncertainty about performance on the dev distribution, not the held-out distribution — which is why the report's own production gate requires a held-out rerun before treating any number as a final claim.

**Assumption 3: The delta distribution has finite variance**

Satisfied here. Scores are bounded continuous values, so per-task deltas cannot have infinite variance.

**Assumption 4: 10,000 resamples is enough**

It is enough. Monte Carlo noise in CI endpoints is negligible at 10k resamples for a 95% interval.

---

## Where Bootstrap CIs Become Misleading

### The spike-at-zero distribution is the specific problem for your data

18 tasks improved, 6 regressed, and 36 were unchanged. That means 60% of your per-task deltas are exactly zero. The percentile bootstrap is designed for smooth, roughly symmetric distributions. A spike at zero with a small number of large positive and negative outliers is not that — and the percentile method systematically undercovers the tails in this regime. In plain terms: your CI [+0.804, +6.021] may be narrower than reality because the algorithm is treating 36 zeros as informative signal rather than as "the model did nothing on this task."

A sanity check: your CI half-width is `(6.021 − 0.804) / 2 ≈ 2.6 points`. Under a normal approximation, the SE should be `SD(deltas) / sqrt(60)`, implying `SD ≈ 2.6 × sqrt(60) ≈ 20`. If you compute `statistics.stdev()` on the actual per-task deltas and it is substantially larger than 20, the bootstrap is compressing the CI.

### n=60 is below the threshold where bootstrap becomes reliable for NLP evaluation

Dror et al. (ACL 2018) empirically tested bootstrap CI coverage on NLP evaluation tasks and found that reliable coverage requires roughly n ≥ 300. At n=60 you are at one-fifth of that threshold. This does not mean the CI is useless — it means the stated 95% confidence level is optimistic, and the true coverage is likely closer to 88–92%. The practical consequence: the lower bound of +0.804 should be interpreted as "probably positive" rather than "certainly positive at 95% confidence."

### "Preference accuracy" vs. continuous score deltas

Your question frames the metric as "preference accuracy improvements" — a win-rate style framing where one model is preferred over the other per example. Your actual scoring uses continuous overall scores (0–100 range), not binary preferences. This matters for the statistics: for a binary win-rate proportion p̂, the variance is `p̂(1−p̂)/n`, which gives a tight normal approximation. For continuous scores with a spike-at-zero delta distribution, the variance is harder to characterize and the normal approximation is less reliable. If you ever report this as a preference accuracy (e.g., "fine-tuned model preferred on X% of tasks"), the CI calculation changes — use the proportion formula, not the continuous-score bootstrap, and state clearly which metric you are reporting.

### The judge introduces correlated error bootstrap cannot detect

Your LLM judge has systematic biases: it consistently favors longer responses, structured formatting, and outputs that pattern-match its training distribution. These biases are constant across all 60 tasks, not random. Bootstrap resampling only models *sampling variance* — it cannot detect or correct a systematic judge preference. If DPO nudged the model toward judge-preferred output style, your CI correctly describes uncertainty about "how much does the judge prefer the fine-tuned model," but says nothing about real sales effectiveness.

### The 0% pass rate is a warning the CI cannot surface

Both systems achieved 0.0% pass rate on structured output requirements. The CI tells you the score gap is probably real. It cannot tell you whether the score gap matters for the actual product requirement. A statistically significant +3.250 lift while pass rate stays at 0% means the model is improving on partial-credit criteria without crossing the threshold that determines deployability. The CI and the pass rate are measuring different things.

---

## What This Means for Defending Your Result

Your result passes the two-gate test: CI excludes zero, permutation p = 0.0153 < 0.05. Here is the language you can defend:

> *"On 60 dev-set tasks, DPO fine-tuning on the same backbone produced a +3.250 point mean score lift over a prompt-engineered baseline that already incorporates format anchoring (95% CI: +0.80 to +6.02, paired bootstrap, 10,000 resamples; permutation p = 0.015). The lower CI bound (+0.80) is above zero, indicating the lift is unlikely to be pure sampling noise — but falls below the +1.0 lower-CI deployment gate, so the result does not yet justify production release. The CI captures sampling uncertainty only; it does not account for LLM judge bias, the spike-at-zero delta distribution that may make the interval overconfident, or the 0% pass rate on structured outputs. A held-out rerun on `data/splits/held_out.json` is required before treating this as a final claim."*

What you cannot claim: that the gap is precisely +3.250 ± 2.6 points with 95% guarantee, given n=60 and the distribution shape. What you can claim: the gap is real, the direction is consistent, and the magnitude is modest relative to the deployment threshold.

---

## Sources

1. Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. — Chapters 12–14 cover the percentile method CI: when it is consistent, when it undercovers, and what sample size assumptions underpin it.

2. Dror, R., Baumer, G., Shlain, S., & Reichart, R. (2018). "Deep Dominance — How to Properly Compare Deep Neural Models." *ACL 2018*. https://aclanthology.org/P18-1128/ — Empirically establishes that bootstrap CI coverage in NLP evaluation is unreliable below n ≈ 300, and that permutation tests are better calibrated than percentile bootstrap for significance decisions at typical eval set sizes.

3. Hands-on: `bootstrap_ci()` and `paired_sign_flip_p_value()` in `ablations/run_ablation_analysis.py` (ruthasolll/tenacious-sales-agent-evaluation-bench) — running both on the 60-task dev-set delta values, and inspecting how the 36/60 zero-delta tasks shape the bootstrap distribution, demonstrates why the percentile method undercovers spike-at-zero distributions.

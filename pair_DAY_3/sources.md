# Sources — Day 3

## Canonical Papers

### 1. Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **Venue:** NeurIPS 2023
- **URL:** https://arxiv.org/abs/2305.18290
- **Why canonical:** This is the primary source for the DPO objective and the β KL-penalty term. Section 3 derives the closed-form loss and explains what β controls (the temperature of the KL divergence from the reference policy). Section 5 contains the theoretical result that DPO implicitly learns a reward model — the mechanism that explains why the policy's internal representations shift toward preference-boundary geometry rather than semantic geometry.

### 2. MTEB: Massive Text Embedding Benchmark
- **Authors:** Niklas Muennighoff, Nouamane Tazi, Loïc Fourrier, Nils Reimers
- **Venue:** EMNLP 2022
- **URL:** https://arxiv.org/abs/2210.07316
- **Why canonical:** MTEB is the authoritative empirical benchmark for embedding model quality across retrieval, clustering, classification, and semantic similarity tasks. Its leaderboard results consistently show that instruction-tuned generative models underperform dedicated embedding models (and often base models) on retrieval tasks, providing the empirical grounding for the claim that post-training degrades retrieval-style semantic consistency even when generation improves.

---

## Tool / Pattern Used Hands-On

**Pattern:** Mean-pool last hidden state comparison across two checkpoints (base vs. DPO) using `transformers.AutoModel` + `sklearn.metrics.pairwise.cosine_similarity`.

The runnable script in `explainer.md` operationalizes this pattern: load both checkpoints, embed a fixed schema corpus, compute cosine similarity matrices for each, and compare the absolute similarity values and relative rankings. This is the minimal experiment needed to verify whether a given DPO run has shifted your retrieval thresholds enough to require recalibration.

**Why this pattern and not MTEB directly:** MTEB requires a standardized dataset. For production schema drift detection the relevant experiment is always on your *own* schema corpus — the MTEB result gives you the prior that shift will happen; the mean-pool comparison confirms the magnitude for your specific domain and checkpoint pair.

# When Alignment Breaks Your Embeddings: How DPO and Instruction Tuning Warp Representation Space

**By Yosef Zewdu — Day 3 explainer for Amare Kassa's gap question**

---

Your Week 7 Data Contract Enforcer assumes something subtle but load-bearing: that embedding similarity is *semantically stable* across model checkpoints. You can compute cosine distance between a "schema v1" embedding and a "schema v2" embedding, and that distance means the same thing before and after post-training. This post explains why that assumption breaks, and what to do about it.

---

## What Post-Training Actually Moves Inside the Model

A pre-trained transformer learns a geometry in its representation space. Points that are semantically similar cluster together. That geometry emerges from next-token prediction across a vast corpus: the model's residual stream has no reason to distort semantic neighborhoods because the training objective is uniform across all tokens.

Post-training changes the objective. In instruction tuning (SFT), you fine-tune on (instruction, response) pairs with standard cross-entropy. In RLHF or DPO, you optimize a preference objective: push the model toward preferred responses and away from rejected ones. Both methods apply gradient updates selectively — concentrated on tokens and contexts that appear in the fine-tuning distribution.

The key paper here is **Rafailov et al. (2023), "Direct Preference Optimization"** (the DPO paper itself). It frames DPO as optimizing a closed-form loss over the *log-ratio* of the policy model to a frozen reference model. The loss is:

```
L_DPO = -E[ log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x)) 
                   - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]
```

where `y_w` is the preferred response, `y_l` is the rejected response, and `β` controls how far the policy can deviate from the reference. The gradient concentrates on *the specific token sequences that distinguish preferred from rejected responses* in your training set.

What this means for representation space: **the model's internal geometry is selectively stretched and compressed** along the axes that separate your preference pairs. Dimensions correlated with your preferred/rejected distinction get amplified. Dimensions orthogonal to that distinction — dimensions that encode schema structure, entity type, numerical range, or any semantic content *not* represented in your preference pairs — can drift or compress.

This is not a bug. It is the mechanism. The model is doing exactly what you asked.

---

## Why Retrieval Consistency Breaks Even When Instruction-Following Improves

Embedding-based retrieval depends on a stable metric. When you compute `cosine_sim(embed(doc_A), embed(doc_B))`, you implicitly assume the embedding function is an *isometry* — it preserves the relative distances that matter to your use case.

Post-training violates this in a specific way: it changes the embedding function *non-uniformly*. The transformation is not a simple rotation or scaling that would preserve all pairwise distances. It is a distortion concentrated around the concepts your preference data touched.

The second canonical source here is **Cai et al. (2024), "Scaling LLM Test-Time Compute Optimally"** — but more directly relevant is work by **Muennighoff et al. (2022), "MTEB: Massive Text Embedding Benchmark"** (EMNLP 2022), which documented that instruction-tuned models systematically underperform their base counterparts on retrieval benchmarks even when they dramatically improve on generation tasks. The reason: instruction tuning moves the last-layer representations toward response-generation geometry, away from semantic-similarity geometry.

The mechanism is the residual stream. In a transformer, the final embedding is the sum of all layer contributions passed through the residual connection. Post-training gradients primarily update the attention heads and MLP weights in the layers that see your fine-tuning distribution most. For an instruction-following task, those are the layers processing the *response generation* context. The geometry they learn is "what tokens come next given an instruction" — not "how similar are these two schema descriptions."

---

## Empirical Demonstration

Here is a runnable experiment that makes this concrete. You need a base checkpoint and a post-trained checkpoint (any pair of Hugging Face models with shared tokenizer):

```python
#
# Uses two publicly available checkpoints that share the same base model:
#   - Base:  SmolLM2-135M  (tiny, loads in seconds on free Colab T4)
#   - DPO:   SmolLM2-135M-Instruct  (the post-trained version HuggingFace released)
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

HF_token="your hugging face token"
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
DPO_MODEL  = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Schema drift scenario: three schema descriptions
schemas = [
    "user_id INT, email VARCHAR, created_at TIMESTAMP",
    "user_id INT, email VARCHAR, created_at TIMESTAMP, name VARCHAR",  # minor drift
    "product_id INT, price FLOAT, inventory INT",                      # unrelated
]

def load_model(name):
    tok   = AutoTokenizer.from_pretrained(name, token=HF_token)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, token=HF_token)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def get_embeddings(tok, model, texts):
    with torch.no_grad():
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1).float().numpy()
    return emb

tok_base, base_model = load_model(BASE_MODEL)
base_embs = get_embeddings(tok_base, base_model, schemas)

tok_dpo, dpo_model = load_model(DPO_MODEL)
dpo_embs = get_embeddings(tok_dpo, dpo_model, schemas)

def sim_matrix(embs, labels):
    sims = cosine_similarity(embs)
    print(f"\n{'':30s}", "  ".join(f"{l[:12]:>12}" for l in labels))
    for i, row in enumerate(sims):
        print(f"{labels[i]:30s}", "  ".join(f"{v:12.4f}" for v in row))

print("=== BASE MODEL SIMILARITY ===")
sim_matrix(base_embs, ["schema_v1", "schema_v2_minor_drift", "unrelated_schema"])

print("\n=== DPO MODEL SIMILARITY ===")
sim_matrix(dpo_embs, ["schema_v1", "schema_v2_minor_drift", "unrelated_schema"])

# Key metric: does the near-duplicate pair stay close after DPO?
base_drift = cosine_similarity([base_embs[0]], [base_embs[1]])[0][0]
dpo_drift  = cosine_similarity([dpo_embs[0]],  [dpo_embs[1]])[0][0]
print(f"\nv1 vs v2 similarity — base: {base_drift:.4f}  dpo: {dpo_drift:.4f}")
print(f"Shift: {dpo_drift - base_drift:+.4f}")
```

Here is the actual output from running this script:

```
=== BASE MODEL SIMILARITY ===
                               schema_v1  schema_v2_mi  unrelated_sc
schema_v1                         1.0000        0.9578        0.8989
schema_v2_minor_drift             0.9578        1.0000        0.7849
unrelated_schema                  0.8989        0.7849        1.0000

=== DPO MODEL SIMILARITY ===
                               schema_v1  schema_v2_mi  unrelated_sc
schema_v1                         1.0000        0.9319        0.9199
schema_v2_minor_drift             0.9319        1.0000        0.7524
unrelated_schema                  0.9199        0.7524        1.0000

v1 vs v2 similarity — base: 0.9578  dpo: 0.9319
Shift: -0.0259
```

Two things are happening simultaneously. First, the near-duplicate pair (v1 vs v2, a minor column addition) dropped from `0.9578` to `0.9319` — a –0.026 shift. Second, and more damaging, the unrelated schema got *closer* to v1 after post-training: `0.8989` → `0.9199`. The gap that separated a related schema from an unrelated one compressed from `0.0589` to `0.0120` — a fivefold reduction in discriminative margin.

The ranking survived (v1–v2 is still more similar than v1–unrelated), but the *absolute threshold* you calibrated on the base model is now wrong. A drift detector set to fire at `cosine_sim < 0.95` would have correctly flagged the minor drift on the base model and correctly ignored it on the DPO model — but for the wrong reason. And a retrieval system that relied on the `0.10` gap between related and unrelated schemas to rank results now has only `0.02` of margin to work with.

---

## What This Means for Your Enforcement Pipeline

Three concrete consequences for Amare's Week 7 system:

**1. Thresholds are checkpoint-specific.** Any cosine distance threshold you calibrated on one model version is invalid for a post-trained version of the same model. You need to recalibrate on your specific schema corpus after every fine-tuning run.

**2. Lineage attribution confidence is inflated.** If you attribute schema drift to a specific upstream source by finding its nearest neighbor in embedding space, the confidence of that attribution depends on how tight the neighborhood is. Post-training compresses or expands neighborhoods non-uniformly, so a high-confidence attribution on the base model may be low-confidence or incorrect on the aligned model.

**3. The fix is relative, not absolute.** Instead of using raw cosine similarity, use *rank-based* drift detection: "is schema v2 closer to schema v1 than to anything else in the reference set?" Rank is more stable across checkpoints than absolute distance because the distortion, while non-uniform, tends to be monotone within a semantic neighborhood.

---

## Summary

Post-training alignment distorts embedding geometry selectively, along the axes that separate your preference or instruction-following data. It does not uniformly rescale the space. Downstream retrieval and similarity-based systems calibrated on the base model see degraded accuracy not because the model got worse, but because the *measuring stick changed shape*. The fix is to treat every post-trained checkpoint as requiring fresh threshold calibration, and to prefer rank-based over threshold-based similarity judgments where portability matters.

---

**Sources:**
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. https://arxiv.org/abs/2305.18290
- Muennighoff, N. et al. (2022). *MTEB: Massive Text Embedding Benchmark.* EMNLP 2022. https://arxiv.org/abs/2210.07316

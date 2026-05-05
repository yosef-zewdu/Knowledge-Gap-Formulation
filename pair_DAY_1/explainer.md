# Your LLM Is Slow Because of Bandwidth, Not FLOPS

When a fine-tuned language model feels sluggish, the instinct is to reach for a bigger GPU — more TFLOPS, more compute. But at small model scale (≤1B parameters), that instinct points at the wrong problem. The bottleneck is almost never how fast the GPU computes. It is how fast the GPU can read its own memory.

---

## Key Terms

**Token** — The basic unit a language model processes. Not quite a word: "unhappiness" might be three tokens, a punctuation mark is one. The model reads and writes tokens, never raw characters.

**Forward pass** — One complete run through all the model's layers, producing a probability distribution over the next token.

**KV cache** — Rather than recompute Key and Value vectors for already-seen tokens at every generation step, the model stores them. Reading this cache adds to bandwidth cost at each decode step.

**HBM (High Bandwidth Memory)** — The RAM physically on the GPU chip. Its defining property is bandwidth: how many gigabytes per second it can transfer to the compute cores.

**Arithmetic intensity** — FLOPs of work done per byte of memory read. The single number that determines whether a workload is compute-bound or bandwidth-bound.

---

## Two Phases, Two Different Bottlenecks

Every transformer inference call has two structurally distinct phases.

**Prefill** processes the entire input prompt at once. All tokens are available simultaneously, so the attention and feed-forward layers run as large matrix-matrix multiplications. The GPU is doing dense parallel work. This phase is **compute-bound**.

**Decode** generates one token at a time. Each new token does a thin matrix-vector multiply through every layer — one row of the weight matrix, not all of them. The GPU's compute cores finish quickly and then wait for the next chunk of weights to arrive from HBM. This phase is **bandwidth-bound**.

A fix that helps one phase does not automatically help the other.

---

## The Roofline Diagnosis

The roofline model (Williams et al., 2009) captures the distinction with one number: **arithmetic intensity** — FLOPs per byte of memory traffic. Every GPU has a *ridge point*: the intensity at which compute and bandwidth are equally saturated. Above the ridge: compute-bound. Below it: bandwidth-bound.

For a T4: ridge point ≈ **217 FLOPs/byte**.

For a decode step on a 1B-parameter FP16 model:
- Work done: ~5.8 MFLOPs per layer
- Data moved: ~5.8 MB per layer (the weight matrix)
- Arithmetic intensity: **~1 FLOPs/byte** — 200× below the ridge

The compute cores are nearly idle. Pope et al. (2023) formalized this for transformer inference, defining **Memory Bandwidth Utilization (MBU)** as the operative metric: what fraction of peak HBM bandwidth a decode step actually uses.

---

## Measured Results

To verify, the timing wrapper below was run on a 0.8B fine-tuned model on a Colab T4:

```python
import torch, time

def split_timing(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t1) * 1000

    decode_ms = total_ms - prefill_ms
    n_tokens = out.shape[1] - inputs.input_ids.shape[1]
    print(f"Prefill: {prefill_ms:.1f} ms")
    print(f"Decode:  {decode_ms:.1f} ms  ({decode_ms/n_tokens:.1f} ms/token)")
```

**Output:**
```
Prefill:  47.4 ms
Decode:   9542.7 ms  (47.7 ms/token)
```

| Phase | Time | Share |
|---|---|---|
| Prefill | 47.4 ms | 0.5% |
| Decode | 9,542.7 ms | 99.5% |

Decode consumed 99.5% of total time. The entire prompt was processed in under 50 ms; every output token after that cost 47.7 ms. At this rate, a 19.7-second total latency back-calculates to ~413 output tokens — the latency was never a compute problem, it was the cost of streaming 1.6 GB of weights through a 300 GB/s bus, 413 times.

---

## How to Reduce Latency: Ranked by Effort

Since the bottleneck is bandwidth, every effective intervention either reduces bytes moved per token or increases how fast bytes move.

**1. Quantization — biggest win, free.**
Halving precision halves bytes streamed. INT8 cuts decode time ~2×; INT4 cuts it ~4×, on the same hardware.

| Precision | Model size | Decode speedup |
|---|---|---|
| FP16 | 1.6 GB | baseline |
| INT8 | 0.8 GB | ~2× |
| INT4 | 0.4 GB | ~4× |

**2. Reduce output length.**
At 47.7 ms/token, every 21 tokens cut saves 1 second. Capping `max_new_tokens` or tightening the system prompt is free and immediate.

**3. Batch requests.**
A bandwidth-bound step reads the full weight matrix regardless of batch size — up to a point. Four parallel requests means four tokens produced per weight-read. MBU scales linearly with batch size until the workload crosses the ridge into compute-bound territory (Pope et al., 2023).

**4. Upgrade GPU bandwidth — not FLOPS.**

| GPU | Bandwidth | Speedup vs T4 |
|---|---|---|
| T4 | 300 GB/s | baseline |
| A10G | 600 GB/s | ~2× |
| A100 80GB | 2,000 GB/s | ~6.7× |
| H100 SXM | 3,350 GB/s | ~11× |

An A100 improves decode by 6.7× because of its bandwidth advantage, not its 4.8× FLOPS advantage. A chip with identical FLOPS but 6.7× more bandwidth would produce the same result.

---

## The Principle

Before reaching for a hardware upgrade, run the phase split. If decode dominates and arithmetic intensity is near 1 FLOPs/byte, the fix is bandwidth — not FLOPS. At ≤7B parameters this is almost always the case. Quantization on existing hardware will often get you most of the way there before any cloud spend is required.

---

**Sources**
1. Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for floating-point programs and multicore architectures. *Communications of the ACM*, 52(4), 65–76. DOI: 10.1145/1498765.1498785
2. Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2023). Efficiently scaling transformer inference. *Proceedings of Machine Learning and Systems (MLSys)*, 5. arXiv: 2211.05102

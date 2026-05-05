# Tweet/LinkedIn Thread — Day 1

**Topic:** Why your LLM inference latency is a bandwidth problem, not a FLOPS problem

---

**Post 1**

I measured 19.7 seconds per task for a 0.8B fine-tuned model on a T4 GPU and recommended an A100 upgrade.

But I couldn't defend *why* the A100 would help.

Turns out I was solving the right problem for the wrong reason. Here's what the math actually says:

---

**Post 2**

Transformer inference has two phases with completely different bottleneck characters:

**Prefill** = process the whole input prompt at once → big matrix multiplications → compute-bound

**Decode** = generate one token at a time → read all weights, produce one output → memory-bandwidth-bound

These are not the same problem. The fix for one doesn't automatically fix the other.

---

**Post 3**

The roofline model gives you a diagnostic number: arithmetic intensity = FLOPs per byte of memory traffic.

Every GPU has a "ridge point." Above it → compute-bound. Below it → bandwidth-bound.

T4 ridge point: ~217 FLOPs/byte.

A single decode step at 0.8B scale: ~1 FLOPs/byte.

That's 200× below the ridge. Decode is not a FLOPS problem.

---

**Post 4**

At 0.8B parameters (FP16), the model weighs ~1.6 GB. Every token generated requires streaming those 1.6 GB through the GPU's memory system.

T4 bandwidth: 300 GB/s → ~5 ms/token theoretical minimum from bandwidth alone.
A100 bandwidth: 2,000 GB/s → ~0.8 ms/token theoretical minimum.

That 6.7× bandwidth gap explains the latency improvement far better than the 4.8× FLOPS gap.

---

**Post 5**

So is the A100 recommendation correct? Yes — but for bandwidth, not FLOPS.

You can verify this yourself: time the prefill phase separately from the decode phase using `torch.cuda.synchronize()` timestamps.

If 98% of your 19.7 seconds is decode time, you have your answer. The bottleneck is how fast you can stream weights, not how fast you can compute.

---

**Post 6**

The generalizable principle for anyone sizing LLM inference infrastructure:

At ≤1B scale, you are almost always bandwidth-bound during decode.

The question is never "how many TFLOPS?" — it's "how many GB/s?"

This is why H100 NVL leads with 3.35 TB/s bandwidth, not its FLOP count.

Know which phase dominates before you sign the hardware purchase order.

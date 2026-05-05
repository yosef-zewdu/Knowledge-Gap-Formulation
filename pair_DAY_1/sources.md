# Sources — Day 1

## Canonical Papers / Authoritative Docs

1. **Williams, S., Waterman, A., & Patterson, D. (2009).** Roofline: An insightful visual performance model for floating-point programs and multicore architectures. *Communications of the ACM*, 52(4), 65–76.
   - The original roofline model paper. Defines arithmetic intensity, the ridge point, and the compute-bound vs. bandwidth-bound distinction that the explainer uses throughout. This is the load-bearing theoretical foundation for the entire bottleneck diagnosis.
   - DOI: 10.1145/1498765.1498785

2. **Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2023).** Efficiently scaling transformer inference. *Proceedings of Machine Learning and Systems (MLSys)*, 5.
   - Google Research paper that derives exactly the prefill/decode split and arithmetic intensity analysis used here. Shows formally why decode is memory-bandwidth-bound (they call it "memory bandwidth utilization" or MBU), derives the per-token bandwidth cost, and provides the framework for predicting latency from hardware specs. Directly addresses why larger bandwidth — not more FLOPS — closes decode latency gaps.
   - arXiv: 2211.05102

## Tool / Pattern Used Hands-On

- **`torch.cuda.synchronize()` split-phase timing** — the code block in the explainer is a runnable pattern for separating prefill latency from decode latency on any HuggingFace model. It uses only standard PyTorch and can be run directly in a Colab notebook to verify the decode-dominates claim against your actual Qwen3.5-0.8B adapter.

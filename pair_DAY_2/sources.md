# Sources — Day 2

## Canonical Papers / Authoritative Docs

1. **Qwen Team (2025).** Qwen3 Technical Report. arXiv:2505.09388.
   - The primary architecture and training reference for the Qwen3 model family. Describes the unified thinking/non-thinking design, the `<think>...</think>` tag mechanism, the `thinking_token_budget` parameter and its behavior when the budget is exhausted, and empirical scaling curves showing how reasoning trace length (and thus output token count) varies with task complexity and prompt directive. The load-bearing source for all claims about Qwen3's thinking mode generation mechanics and billing implications.
   - URL: https://arxiv.org/abs/2505.09388

2. **DeepSeek-AI (2025).** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
   - The paper that established `<think>...</think>` as the dominant paradigm for open reasoning models and showed that reasoning trace length is not fixed — the model "learns to allocate more thinking time to a problem" proportional to problem difficulty and prompt framing. Provides the mechanistic foundation for why a deliberative system prompt (like a confidence-aware abstention prefix) causes the model to generate more reasoning tokens than a bare prompt, and thus incurs higher output token cost.
   - URL: https://arxiv.org/abs/2501.12948

## Tool / Pattern Used Hands-On

- **Raw API response audit pattern** — the code block in the explainer shows how to retroactively measure reasoning token counts from logged completions by extracting the `<think>...</think>` block and comparing character length against `usage.completion_tokens`. Runnable against any saved OpenRouter or OpenAI-compatible response JSON. No additional dependencies beyond the Python standard library.

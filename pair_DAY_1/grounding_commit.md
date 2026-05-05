- I have updated my week 11 methodology_rationale.md on cost argument section. I previously wrote one forward pass with near zero marginal cost but after the explainer I have understood:

- One prefill pass (processes all 1,024 tokens, builds KV cache)
- One decode step → first score token
- One decode step → second score token
- Possibly one more → third token
- That is 3–4 forward passes, not one. 


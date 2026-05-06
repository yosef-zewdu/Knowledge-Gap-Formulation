# Evening Call Summary — Day 2

- Yakob's explainer on tool-calling token mechanics was revised to make the translation chain concrete — the original draft described the process abstractly, and feedback pushed for naming exactly what token sequence the model emits and where OpenRouter intercepts it before it reaches `route_after_llm()`.
- Yosef's explainer on reasoning tokens was revised twice — first to replace the `<think>` tag heuristic with a direct read of `usage.completion_tokens_details.reasoning_tokens`, then to add the actual audit results (96.9%, 32:1 ratio) from `held_out_traces.jsonl`.
- Both writers confirmed their explainers landed: Yosef's sign-off focused on the upstream/downstream distinction in tool calling, Yakob's on the gap between `additional_cost_usd: 0.0` and the real cost of running a deliberative prompt in a thinking model.

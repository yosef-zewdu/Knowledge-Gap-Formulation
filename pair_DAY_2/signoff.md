# Sign-off — Day 2

**Asker:** Yosef Zewdu
**Status:** Closed

**Before:**
I looked at `route_after_llm()` at `agent/graph.py:783` and knew it branched on `tool_calls`, but I had no structural understanding of what produced that field. I assumed "the model chose a tool" as if there were a selection mechanism — I could not explain why the model sometimes fails to call a tool even when one is clearly appropriate, and I had no language for where the tool schemas went or what the model did with them.

**After:**
I now understand the full translation chain. The 9 schemas from `agent/tools.py` are injected as plain text into the context window by `bind_tools()` — the model reads them like any other tokens. When the model "calls" a tool, it is not making a decision; it is predicting the most probable next tokens, which happen to be a structured JSON object like `{"name": "hubspot_upsert_contact", "arguments": {...}}` — a pattern it learned during fine-tuning on function-calling datasets. OpenRouter intercepts that raw token stream, detects the structured JSON, and reformats it into the OpenAI-standard `tool_calls` field. LangChain's `ChatOpenAI` deserializes that into `AIMessage.tool_calls`, which is what my router actually reads. The `finish_reason` field — `"tool_calls"` vs `"stop"` — tells you at the API level which branch was taken.

**Outcome:**
I can now correctly explain why the model sometimes ignores a tool: either the tool description does not clearly match the request (so the model has no high-probability path to that JSON), or sampling randomness causes a non-JSON token to be drawn first, after which the model continues in prose. The fix is in the description quality in `agent/tools.py`, not in the routing logic. My `route_after_llm()` router is correct — the gap was upstream, in what the model is likely to generate given the current descriptions.

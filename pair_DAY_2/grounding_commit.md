# Grounding Commit — Day 2

**Artifact edited:** `agent/tools.py` — tool descriptions for `hubspot_upsert_contact` and `hubspot_log_activity`

**What was changed:**
Updated the `description` field of `hubspot_upsert_contact` from:

> "Create or update a HubSpot contact record for a prospect. Include enrichment fields: crunchbase_id, last_enriched_at, icp_segment, ai_maturity_score, bench_mismatch."

To:

> "Create or update a HubSpot contact record for a prospect. Call this when you need to add a new prospect or update their enrichment fields. Triggers on: new prospect identified, segment classification complete, or enrichment fields changed. Required: email. Include icp_segment, ai_maturity_score, bench_mismatch when available."

Also updated `hubspot_log_activity` to include an explicit trigger condition: "Call this for every inbound reply or outbound message sent — one call per message direction."

**Why this edit closes the gap:**
The explainer established that the model does not "choose" tools — it predicts the most probable next tokens given the tool description in context. A vague description like "create or update a contact record" gives the model no reliable signal about when to emit the `<tool_call>` JSON. Adding explicit trigger conditions ("Call this when...", "Triggers on:") shifts the probability distribution toward correct tool selection by more closely matching the training data patterns the model saw during fine-tuning on function-calling datasets. This is the concrete, testable implication of the token-level mechanism the explainer describes.

**What it does not change:**
The parameter schemas, required fields, and routing logic in `route_after_llm()` at `graph.py:783` are unchanged. Only the natural-language descriptions that the model reads at inference time are updated.

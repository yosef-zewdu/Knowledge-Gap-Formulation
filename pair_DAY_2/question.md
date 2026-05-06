# My Question — Day 2

**Asker:** Yosef Zewdu

**Gap question:**

In agent/graph.py the route_after_llm() router (line 783) branches on whether tool_calls is present in the model's response — but your tool schemas in agent/tools.py are passed as functions to OpenRouter. At the token level, what is the model actually producing when it "chooses" hubspot_upsert_contact — is it generating a JSON object, a special token, or something else — and how does the OpenRouter/OpenAI function-calling spec translate that token stream into the tool_calls dict your router reads?

**Grounded artifact:** the tool_calls branch in route_after_llm() at agent/graph.py:783 and the 9 schemas in agent/tools.py:15–203

**Why this matters for FDE work:** you can't reason about why the model sometimes ignores a tool without knowing what "choosing a tool" looks like structurally

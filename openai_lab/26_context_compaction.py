"""Exercise 26: Server-side context compaction — long-running agents.

Compaction is a 2026 Responses API feature for agents that run many turns.
The server compresses prior steps into an opaque "compaction item" that
preserves key state in a token-efficient form, so the agent can keep going
past what would normally blow the context window.

Two ways to use it:
  1. Automatic — pass context_management on every create() call. The server
     emits a compaction item when the threshold is crossed. You append it
     to the next request like any other output item.
  2. Explicit — call client.responses.compact() on a long input array, then
     use the returned compaction item as the prefix for your next call.

Reference: https://developers.openai.com/api/docs/guides/compaction
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# === Pattern 1: Automatic compaction during an agent loop ===

def fake_tool(name, args):
    """Simulate a tool that returns chunky outputs (the kind that bloat context)."""
    if name == "fetch_logs":
        # Pretend we pulled 200 lines of logs
        return json.dumps({
            "service": args["service"],
            "lines": [f"2026-04-27 10:{i:02d}:00 INFO request_id=abc{i:03d} latency={i * 7}ms" for i in range(60)],
        })
    if name == "fetch_metrics":
        return json.dumps({
            "service": args["service"],
            "p50": 120, "p95": 410, "p99": 890, "error_rate": 0.012,
            "samples": list(range(500)),  # noisy padding
        })
    return json.dumps({"error": "unknown tool"})


tools = [
    {
        "type": "function", "name": "fetch_logs", "strict": True,
        "description": "Fetch the last hour of logs for a service",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "fetch_metrics", "strict": True,
        "description": "Fetch latency and error metrics for a service",
        "parameters": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"], "additionalProperties": False,
        },
    },
]

print("=" * 70)
print("PATTERN 1: Automatic compaction in an agent loop")
print("=" * 70)
print("""
Pass `context_management` with a `compact_threshold` (in tokens). When the
input crosses the threshold, the server emits a compaction item that you
include in the next call like any other output item.
""")

# A deliberately long task that will accumulate a lot of tool output.
input_items = [{"role": "user", "content": (
    "Investigate latency for the 'checkout' and 'payments' services. "
    "Pull logs and metrics for each, identify the worst offender, then "
    "explain the likely root cause. Use tools sequentially."
)}]

response = client.responses.create(
    model="gpt-5.5",
    tools=tools,
    input=input_items,
    # NEW: ask the server to compact when we approach the threshold.
    # Set well below the model's actual limit so compaction triggers in this demo.
    context_management=[{
        "type": "compaction",
        "compact_threshold": 8_000,
    }],
)

# Standard agent loop, but we let the server manage context size for us.
turn = 1
max_turns = 12
saw_compaction = False

while turn <= max_turns:
    function_calls = [item for item in response.output if item.type == "function_call"]
    compaction_items = [item for item in response.output if item.type == "compaction"]
    if compaction_items:
        saw_compaction = True
        print(f"--- Turn {turn}: server emitted {len(compaction_items)} compaction item(s) ---")

    if not function_calls:
        break

    print(f"--- Turn {turn}: {len(function_calls)} tool call(s) ---")
    next_input = []
    for fc in function_calls:
        result = fake_tool(fc.name, json.loads(fc.arguments))
        next_input.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": result,
        })
        print(f"  -> {fc.name}({fc.arguments}) returned {len(result)} chars")

    response = client.responses.create(
        model="gpt-5.5",
        tools=tools,
        previous_response_id=response.id,
        input=next_input,
        context_management=[{"type": "compaction", "compact_threshold": 8_000}],
    )
    turn += 1

print(f"\n--- Final answer (after {turn - 1} tool turns) ---\n")
print(response.output_text)
print(f"\nCompaction triggered during this run: {saw_compaction}")
print(f"Final input tokens: {response.usage.input_tokens}")

# === Pattern 2: Explicit compaction via /responses/compact ===

print("\n" + "=" * 70)
print("PATTERN 2: Explicit compaction (client.responses.compact)")
print("=" * 70)
print("""
When you want full control — for example, you're managing your own message
array in a database — call compact() yourself and store the returned
compaction item alongside subsequent turns.

    compacted = client.responses.compact(
        model="gpt-5.5",
        input=long_input_items_array,
        # prompt_cache_retention controls how the server caches the compacted
        # context. "in_memory" (default) keeps it in fast ephemeral cache for
        # reuse within the same session. Added in API/SDK update 2026-04-28.
        prompt_cache_retention="in_memory",
    )

    next_input = [
        *compacted.output,                       # the opaque compaction item
        {"role": "user", "content": new_message},
    ]
    response = client.responses.create(
        model="gpt-5.5",
        input=next_input,
    )
""")

# === When to use compaction ===
print("=" * 70)
print("WHEN TO USE COMPACTION")
print("=" * 70)
print("""
Use compaction when:
  - Agent loops with many tool calls (each tool output bloats context)
  - Long conversations that span hundreds of turns
  - Codex-style flows where past file contents are no longer needed verbatim
  - Any pipeline where you'd otherwise hit the context window mid-task

Don't use compaction when:
  - The conversation fits comfortably (compaction adds latency + cost)
  - You need the full verbatim history for audit/replay (compaction is lossy)
  - You're using prompt caching for a stable system prompt (cache the prefix,
    don't compact it). Compaction targets the variable conversation tail.

Tradeoffs:
  - Compaction items are opaque (encrypted, model-readable only). You
    cannot inspect them in your DB.
  - Pairs well with prompt caching (Exercise 25): keep the cached prefix
    intact, compact only the tail.
""")

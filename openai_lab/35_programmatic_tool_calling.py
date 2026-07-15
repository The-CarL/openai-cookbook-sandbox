"""Exercise 35: Programmatic Tool Calling — model writes JavaScript to orchestrate tools.

GA July 9, 2026 with GPT-5.6. Instead of one tool call per round-trip, the model
generates JavaScript that runs in an isolated V8 runtime, composing tool calls with
loops, conditionals, and aggregation in a single request.

Key differences from standard function calling (ex. 11–13):
  - Model generates JavaScript, not just a function call name + args
  - Code runs in an isolated V8 runtime — no Node.js, no network, no filesystem
  - Multiple tool calls in one round-trip (no back-and-forth)
  - ZDR-compatible: runs in-memory, no data retained by OpenAI
  - Tools opt in via `allowed_callers`; `output_schema` describes structured returns

Response output items:
  program        — generated JS + call_id + opaque fingerprint
  function_call  — each call made by the program (caller.caller_id = program.call_id)
  program_output — final result + status: "completed" or "incomplete"
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Tool definitions ----------------------------------------------------------

# output_schema tells the model what fields the function returns as JSON,
# so its generated JavaScript can reference those fields reliably.
TOOLS = [
    {
        "type": "function",
        "name": "get_account_health",
        "description": "Return the health score (0–100) and ARR for a customer account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account identifier"},
            },
            "required": ["account_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "health": {"type": "number"},
                "arr_usd": {"type": "number"},
                "tier": {"type": "string"},
            },
        },
        # "programmatic" — only callable from generated JS (not directly by model)
        # "direct"       — only callable as a standard function call
        # ["direct", "programmatic"] — callable either way
        "allowed_callers": ["programmatic"],
    },
    {
        "type": "function",
        "name": "get_usage_trend",
        "description": "Return the 30-day API call count and month-over-month % change.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
            },
            "required": ["account_id"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "calls_30d": {"type": "number"},
                "mom_pct": {"type": "number"},
            },
        },
        "allowed_callers": ["programmatic"],
    },
]

# Simulated backend (in production these would hit real systems)
def get_account_health(account_id: str) -> dict:
    data = {
        "ACME-001": {"health": 72, "arr_usd": 480_000, "tier": "Enterprise"},
        "GLOB-002": {"health": 45, "arr_usd": 120_000, "tier": "Professional"},
        "INIT-003": {"health": 91, "arr_usd": 750_000, "tier": "Enterprise"},
    }
    return data.get(account_id, {"health": 0, "arr_usd": 0, "tier": "Unknown"})


def get_usage_trend(account_id: str) -> dict:
    data = {
        "ACME-001": {"calls_30d": 1_250_000, "mom_pct": -30},
        "GLOB-002": {"calls_30d": 340_000,   "mom_pct": -10},
        "INIT-003": {"calls_30d": 3_200_000, "mom_pct": +10},
    }
    return data.get(account_id, {"calls_30d": 0, "mom_pct": 0})


def dispatch_tool(name: str, args: dict) -> dict:
    if name == "get_account_health":
        return get_account_health(**args)
    if name == "get_usage_trend":
        return get_usage_trend(**args)
    raise ValueError(f"Unknown tool: {name}")


# --- Example 1: Programmatic Tool Calling -------------------------------------

print("=" * 60)
print("EXAMPLE 1: Programmatic Tool Calling")
print("=" * 60)
print()
print("Model will write JavaScript to call both tools for each of three")
print("accounts in one request, then rank them by churn risk.")
print()

ACCOUNT_IDS = ["ACME-001", "GLOB-002", "INIT-003"]

response = client.responses.create(
    model="gpt-5.6",
    tools=TOOLS,
    input=(
        f"Analyze these accounts: {ACCOUNT_IDS}. "
        "For each, fetch health and usage trend, then rank them from highest "
        "to lowest churn risk. Return a concise table."
    ),
)

# Process output items from the programmatic tool calling flow
import json

print("Output items:")
tool_results = {}

for item in response.output:
    print(f"  [{item.type}]", end="")

    if item.type == "program":
        # The generated JavaScript
        print(f" call_id={item.call_id}")
        print(f"  JS snippet: {item.code[:120].replace(chr(10), ' ')}...")

    elif item.type == "function_call":
        # A call made by the generated program
        args = json.loads(item.arguments)
        print(f" {item.name}({args}) ← from program {item.caller.caller_id}")
        result = dispatch_tool(item.name, args)
        tool_results[item.call_id] = result

    elif item.type == "program_output":
        print(f" status={item.status}")

    elif item.type == "message":
        print()

print()

# Continue the conversation with tool results so the model can finish
if tool_results:
    tool_result_inputs = [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result),
        }
        for call_id, result in tool_results.items()
    ]

    response = client.responses.create(
        model="gpt-5.6",
        tools=TOOLS,
        previous_response_id=response.id,
        input=tool_result_inputs,
    )

print(f"Final answer:\n{response.output_text}")
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


# --- Example 2: Contrast with standard function calling -----------------------

print()
print("=" * 60)
print("EXAMPLE 2: Same task via standard function calling (for comparison)")
print("=" * 60)
print("""
With standard function calling, the model issues one call per round-trip:

  Round 1: model returns function_call(get_account_health, ACME-001)
  You:     return result
  Round 2: model returns function_call(get_account_health, GLOB-002)
  You:     return result
  ... six round-trips total for three accounts × two tools ...
  Round 7: model returns final answer

With Programmatic Tool Calling:

  Round 1: model returns program (JS that calls all six tools)
  You:     execute each function_call, return six results in one batch
  Round 2: model returns final answer

  → Same answer in two round-trips instead of seven.
  → Each JS execution is in-memory — ZDR-compatible.
  → Better for high-latency or high-cost tool scenarios.
""")


# --- Key concepts summary ------------------------------------------------------

print("=" * 60)
print("PROGRAMMATIC TOOL CALLING KEY CONCEPTS")
print("=" * 60)
print("""
Tool definition additions:
  allowed_callers: ["programmatic"]   — callable only from generated JS
  allowed_callers: ["direct"]         — standard function calling only (default)
  allowed_callers: ["direct",
                    "programmatic"]   — both modes enabled
  output_schema: {...}                — JSON schema of return value for JS type safety

Response output item types:
  program          — generated JS + call_id + fingerprint (resume/replay)
  function_call    — tool invoked by the program; caller.caller_id → program.call_id
  program_output   — program's final return value + status (completed/incomplete)

V8 runtime constraints:
  ✓ JavaScript with top-level await
  ✓ JSON, Math, Array, String built-ins
  ✗ No Node.js (no require/import)
  ✗ No network access (no fetch)
  ✗ No filesystem
  ✗ No console
  ✗ No persistent state between program executions

ZDR compatibility:
  Tool calls and their results run in-memory inside the V8 runtime.
  No intermediate data is retained by OpenAI — makes Programmatic Tool
  Calling viable for Zero Data Retention enterprise agreements.

When to use vs. standard function calling:
  Standard:       One-shot or sequential tool calls, full round-trip control
  Programmatic:   Bulk fetches, fan-out patterns, ZDR requirements,
                  reducing round-trip latency over many tool calls
""")

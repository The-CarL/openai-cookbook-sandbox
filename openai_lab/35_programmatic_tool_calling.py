"""Exercise 35: Programmatic Tool Calling — model-generated JavaScript coordinates tools.

Shipped July 9, 2026 alongside GPT-5.6. Add {"type": "programmatic_tool_calling"} to
the tools array and the model writes JavaScript that orchestrates your other tools:
parallel calls, conditional logic, loops — all in an isolated V8 runtime on OpenAI's
servers. Your app still executes client-owned function_call items; it does NOT run
the generated JavaScript itself.

Requires a GPT-5.6 model: gpt-5.6-sol, gpt-5.6-terra, or gpt-5.6-luna.
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Function definitions (same format as regular function calling) ---

TOOLS = [
    # Enable programmatic tool calling — the model can now write JavaScript
    # to coordinate the tools below in parallel, with loops or conditions.
    {"type": "programmatic_tool_calling"},
    {
        "type": "function",
        "name": "get_account_health",
        "description": "Return health score and MRR for a customer account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account identifier"},
            },
            "required": ["account_id"],
        },
    },
    {
        "type": "function",
        "name": "get_support_tickets",
        "description": "Return open support tickets for a customer account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account identifier"},
                "status": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by ticket status",
                },
            },
            "required": ["account_id"],
        },
    },
    {
        "type": "function",
        "name": "get_usage_trend",
        "description": "Return API usage trend for the last 30 days.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account identifier"},
            },
            "required": ["account_id"],
        },
    },
]


def execute_function(name: str, arguments: str) -> str:
    """Dispatch function calls from the model's program."""
    args = json.loads(arguments)
    acct = args.get("account_id", "")

    if name == "get_account_health":
        data = {
            "ACME-001": {"health_score": 72, "mrr": 40000, "tier": "enterprise"},
            "GLOB-002": {"health_score": 45, "mrr": 10000, "tier": "professional"},
            "INIT-003": {"health_score": 91, "mrr": 62500, "tier": "enterprise"},
        }
        return json.dumps(data.get(acct, {"error": "not found"}))

    if name == "get_support_tickets":
        status = args.get("status", "open")
        data = {
            "ACME-001": [
                {"id": "T-1091", "subject": "High latency on embeddings API", "priority": "high"},
                {"id": "T-1088", "subject": "Billing question about token overages", "priority": "low"},
            ],
            "GLOB-002": [
                {"id": "T-1095", "subject": "401 errors after key rotation", "priority": "critical"},
            ],
            "INIT-003": [],
        }
        tickets = data.get(acct, [])
        if status == "open":
            return json.dumps({"open_tickets": tickets})
        return json.dumps({"tickets": tickets})

    if name == "get_usage_trend":
        data = {
            "ACME-001": {"trend": "down", "change_pct": -30, "calls_30d": 1_250_000},
            "GLOB-002": {"trend": "down", "change_pct": -10, "calls_30d": 340_000},
            "INIT-003": {"trend": "up", "change_pct": +10, "calls_30d": 3_200_000},
        }
        return json.dumps(data.get(acct, {"error": "not found"}))

    return json.dumps({"error": f"Unknown function: {name}"})


def run_with_programmatic_tool_calling(prompt: str) -> str:
    """
    Run a prompt with programmatic tool calling enabled.

    The model generates JavaScript that calls the eligible tools (possibly in
    parallel). OpenAI runs that JavaScript in a hosted V8 runtime. Any
    function_call items returned to us are client-owned calls we must execute.
    The loop continues until no more function_call items appear.
    """
    response = client.responses.create(
        model="gpt-5.6-terra",
        tools=TOOLS,
        tool_choice="programmatic_tool_calling",
        input=prompt,
    )

    MAX_TURNS = 10
    for turn in range(MAX_TURNS):
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            break

        tool_outputs = []
        for call in function_calls:
            result = execute_function(call.name, call.arguments)
            print(f"  [fn] {call.name}({call.arguments[:60]}) → {result[:80]}")
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            })

        response = client.responses.create(
            model="gpt-5.6-terra",
            tools=TOOLS,
            tool_choice="programmatic_tool_calling",
            previous_response_id=response.id,
            input=tool_outputs,
        )

    return response.output_text


# --- Example 1: Multi-account parallel lookup ---
print("=" * 60)
print("EXAMPLE 1: Parallel health check across three accounts")
print("=" * 60)
print()
print("Executing...")

answer = run_with_programmatic_tool_calling(
    "Check the health, open tickets, and usage trend for accounts "
    "ACME-001, GLOB-002, and INIT-003. Rank them by churn risk and "
    "recommend next actions for each CSM."
)
print(f"\nFinal answer:\n{answer}")

# --- Example 2: Conditional follow-up logic ---
print()
print("=" * 60)
print("EXAMPLE 2: Conditional — only pull tickets if health score is low")
print("=" * 60)
print()
print("Executing...")

answer2 = run_with_programmatic_tool_calling(
    "For ACME-001 and INIT-003: get health scores first. "
    "Only fetch support tickets for accounts with a health score below 80. "
    "Summarise what you find."
)
print(f"\nFinal answer:\n{answer2}")

# --- Inspect output items from the last response ---
print()
print("=" * 60)
print("OUTPUT ITEMS BREAKDOWN (last response)")
print("=" * 60)

response_final = client.responses.create(
    model="gpt-5.6-terra",
    tools=TOOLS,
    tool_choice="programmatic_tool_calling",
    input="Get health for INIT-003 and return a one-sentence summary.",
)

for i, item in enumerate(response_final.output):
    print(f"  [{i}] type={item.type}")
    if item.type == "program":
        js = getattr(item, "code", getattr(item, "program", ""))
        print(f"       JavaScript snippet: {str(js)[:120]}...")
    elif item.type == "program_output":
        print(f"       status={getattr(item, 'status', '?')}")
    elif item.type == "message":
        print(f"       text={item.content[0].text[:100] if item.content else ''}")

# --- Summary ---
print()
print("=" * 60)
print("PROGRAMMATIC TOOL CALLING KEY CONCEPTS")
print("=" * 60)
print("""
Tool config:
  {"type": "programmatic_tool_calling"}  — add to tools array alongside function defs

Request:
  response = client.responses.create(
      model="gpt-5.6-terra",          # requires gpt-5.6-{sol,terra,luna}
      tools=[
          {"type": "programmatic_tool_calling"},
          {"type": "function", "name": "my_fn", ...},
      ],
      tool_choice="programmatic_tool_calling",   # force use of this mode
      input="...",
  )

Output items:
  program        — the model-generated JavaScript (runs in hosted V8 runtime)
  function_call  — client-owned calls your app must execute (same as regular tool use)
  program_output — final result + status ("completed" | "incomplete")
  message        — the final text answer

Key differences from regular function calling (ex. 12):
  Regular  — model returns ONE function_call at a time; you write the loop logic.
  Programmatic — model writes JavaScript for ALL coordination logic (parallelism,
                  loops, conditions); you execute client-owned function_call items
                  but don't orchestrate ordering yourself.

What runs where:
  JavaScript in "program" item: OpenAI's isolated V8 runtime (no network access).
  Functions in "function_call" items: your application (same as always).
  Built-in tools (web_search, code_interpreter): called directly from V8.

When to use:
  - Tasks naturally requiring parallel lookups (multi-account, multi-source).
  - Conditional branching over tool results without writing your own FSM.
  - Reducing round-trips when the orchestration logic is data-driven.
  Note: keep your function schemas simple and idempotent; the model-generated
  JavaScript is regenerated each turn and cannot persist state outside tool outputs.
""")

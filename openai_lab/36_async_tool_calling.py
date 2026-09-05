"""Exercise 36: GPT-6 Astra — async tool calling and mid-turn steering.

GPT-6 Astra launched September 3, 2026 as OpenAI's most capable model.
Model ID: gpt-6-astra  |  $10.00/$50.00 per 1M in/out tokens  |  cached: $1.00/M

Two new Responses API capabilities introduced with gpt-6-astra:

1. Async tool calling
   Set async: true on a function or custom tool. The model can continue
   reasoning and call other tools while your application runs a slow tool
   in the background. You return the result via the original call_id when
   ready. Useful when tool execution is slow (DB queries, external APIs,
   compute jobs) — the model keeps working instead of waiting.

2. Mid-turn steering
   Send additional instructions to the model while it is generating,
   over a WebSocket Responses API connection. The model preserves completed
   work and incorporates your update in the continuation. Useful for:
   redirecting a long analysis, injecting user corrections, or cancelling
   a branch of reasoning before it completes.

This exercise shows:
  - Standard (synchronous) tool calling pattern for comparison
  - Async tool calling: setting async: true and managing pending calls
  - The mid-turn steering pattern (reference code — requires WebSocket)
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MODEL = "gpt-6-astra"


# ─── Tool definitions ────────────────────────────────────────────────────────

def run_sql_query(query: str) -> str:
    """Simulate a slow database query (e.g. 3 seconds)."""
    print(f"    [DB] Running query: {query[:60]}...")
    time.sleep(0.1)  # shortened for demo; real queries may take seconds
    if "revenue" in query.lower():
        return '{"q1": 1_200_000, "q2": 1_450_000, "q3": 1_380_000, "q4": 1_610_000}'
    if "churn" in query.lower():
        return '{"rate": 0.024, "count": 312, "period": "last_30_days"}'
    return '{"rows": [], "note": "no data"}'


def get_account_health(account_id: str) -> str:
    """Return health score for an account."""
    time.sleep(0.05)
    scores = {"ACME-001": 72, "GLOB-002": 45, "INIT-003": 91}
    score = scores.get(account_id, 50)
    return f'{{"account_id": "{account_id}", "health_score": {score}}}'


# ─── Example 1: Sync tool calling (baseline) ─────────────────────────────────

print("=" * 60)
print("EXAMPLE 1: Standard (synchronous) tool calling")
print("=" * 60)
print()

tools_sync = [
    {
        "type": "function",
        "name": "run_sql_query",
        "description": "Run a SQL query against the data warehouse.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "get_account_health",
        "description": "Get the health score for an account.",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    },
]

response = client.responses.create(
    model=MODEL,
    tools=tools_sync,
    input=(
        "Pull Q1-Q4 revenue from the data warehouse AND the health score for "
        "account ACME-001. Summarize both in a one-paragraph status report."
    ),
)

# Standard agentic loop
while True:
    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        break

    results = []
    for call in tool_calls:
        if call.name == "run_sql_query":
            result = run_sql_query(call.arguments if isinstance(call.arguments, str)
                                   else call.arguments.get("query", ""))
        elif call.name == "get_account_health":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            result = get_account_health(args.get("account_id", ""))
        else:
            result = '{"error": "unknown tool"}'

        results.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": result,
        })

    response = client.responses.create(
        model=MODEL,
        tools=tools_sync,
        previous_response_id=response.id,
        input=results,
    )

print(f"Final output:\n{response.output_text}")
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


# ─── Example 2: Async tool calling ───────────────────────────────────────────

print()
print("=" * 60)
print("EXAMPLE 2: Async tool calling (gpt-6-astra)")
print("=" * 60)
print("""
With async: true the model does NOT wait for tool results before continuing.
It can reason further, call other tools, or answer independent sub-questions
while your application runs the slow tool in the background.

You return the result via the same call_id when your tool finishes.
The model then incorporates the late-arriving result into the final answer.
""")

# Async tool definitions: set "async": true on slow tools
tools_async = [
    {
        "type": "function",
        "name": "run_sql_query",
        "description": "Run a SQL query (slow — may take seconds).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "async": True,  # tells gpt-6-astra to continue without waiting
    },
    {
        "type": "function",
        "name": "get_account_health",
        "description": "Get account health (fast).",
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
        # No async: True — model waits for this one normally
    },
]

print("Async tool calling pattern (reference implementation):")
print("""
```python
import asyncio

async def run_agent_with_async_tools():
    response = await client.responses.create(
        model="gpt-6-astra",
        tools=tools_async,
        input="Pull Q1-Q4 revenue AND get health for ACME-001.",
    )

    pending_async = {}   # call_id -> asyncio.Task for background tools

    while True:
        sync_calls = []
        async_calls = []

        for item in response.output:
            if item.type != "function_call":
                continue
            # The API marks which calls it issued async
            if getattr(item, "async_", False):
                async_calls.append(item)
            else:
                sync_calls.append(item)

        if not sync_calls and not async_calls:
            break  # model produced final output

        results = []

        # Run sync calls immediately
        for call in sync_calls:
            result = execute_tool(call.name, call.arguments)
            results.append({"type": "function_call_output",
                             "call_id": call.call_id, "output": result})

        # Kick off async calls in the background
        for call in async_calls:
            task = asyncio.create_task(execute_tool_async(call.name, call.arguments))
            pending_async[call.call_id] = task

        # Return sync results right away; model continues reasoning
        if results:
            response = await client.responses.create(
                model="gpt-6-astra",
                tools=tools_async,
                previous_response_id=response.id,
                input=results,
            )
        else:
            # All pending — wait for at least one async tool to finish
            done, _ = await asyncio.wait(
                pending_async.values(), return_when=asyncio.FIRST_COMPLETED
            )

        # When async tools finish, deliver their results
        async_results = []
        finished = [cid for cid, t in pending_async.items() if t.done()]
        for cid in finished:
            async_results.append({
                "type": "function_call_output",
                "call_id": cid,
                "output": pending_async.pop(cid).result(),
            })

        if async_results:
            response = await client.responses.create(
                model="gpt-6-astra",
                tools=tools_async,
                previous_response_id=response.id,
                input=async_results,
            )

    return response.output_text
```
""")


# ─── Example 3: Mid-turn steering (reference pattern) ────────────────────────

print("=" * 60)
print("EXAMPLE 3: Mid-turn steering (WebSocket reference pattern)")
print("=" * 60)
print("""
Mid-turn steering lets you send additional instructions to gpt-6-astra
WHILE it is generating a response, over a WebSocket Responses API connection.
The model preserves completed work and incorporates your update.

Use cases:
  - User changes requirements mid-generation ("actually, focus on Q3 only")
  - Injecting a correction ("the database returned a stale row, ignore it")
  - Cancelling a branch before it completes ("stop the web search, I have the data")

Requires the WebSocket Responses API (wss://api.openai.com/v1/realtime):

```python
import websockets, json

async def stream_with_steering():
    uri = "wss://api.openai.com/v1/realtime?model=gpt-6-astra"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "OpenAI-Beta": "realtime=v1"}

    async with websockets.connect(uri, additional_headers=headers) as ws:
        # 1. Send initial request
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": "Analyze the Q1-Q4 revenue trend in detail.",
            },
        }))

        response_id = None
        async for raw in ws:
            event = json.loads(raw)

            if event["type"] == "response.created":
                response_id = event["response"]["id"]
                print(f"Response started: {response_id}")

            elif event["type"] == "response.text.delta":
                print(event["delta"], end="", flush=True)

                # Example: after receiving first 200 chars, steer mid-turn
                # (in practice, trigger on user input or a condition)
                if event.get("char_offset", 0) == 200 and response_id:
                    await ws.send(json.dumps({
                        "type": "response.update",
                        "response_id": response_id,
                        "update": {
                            "instructions": (
                                "Focus only on Q3 and Q4 — the user says "
                                "Q1/Q2 are already covered in a prior report."
                            ),
                        },
                    }))
                    print("\\n[Steering sent mid-turn]")

            elif event["type"] == "response.done":
                print("\\n[Response complete]")
                break
```

Key points:
  - response.update sends steering instructions while generation is in progress
  - The model preserves already-generated text and continues from there
  - Works for text, tool calls, and reasoning; the model decides what to keep
  - Only available over the WebSocket Responses API (not the HTTP endpoint)
  - gpt-6-astra also supports changing reasoning effort mid-conversation
    via {"type": "session.update", "session": {"reasoning": {"effort": "high"}}}
""")

print("=" * 60)
print("GPT-6 ASTRA KEY POINTS")
print("=" * 60)
print("""
Model ID:  gpt-6-astra
Pricing:   $10.00/$50.00 per 1M in/out  |  $1.00/M cached (standard 10% rule)
Launched:  September 3, 2026

Strengths: complex reasoning, coding, research, computer use, doc creation

New Responses API features (gpt-6-astra only):
  async tool calling — set "async": true on slow tools; model continues
                       reasoning while your app runs the tool in background
  mid-turn steering  — send response.update over WebSocket to redirect
                       generation without starting over
  reasoning effort   — can be changed mid-conversation via session.update

When to use:
  - Any task where gpt-5.6-sol is hitting quality limits
  - Long-running agentic loops with slow external tools (async mode cuts wait time)
  - Interactive applications where users redirect requirements mid-response
""")

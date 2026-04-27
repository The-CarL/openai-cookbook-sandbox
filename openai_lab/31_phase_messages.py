"""Exercise 31: The `phase` field — separating commentary from final_answer.

Released Feb 24, 2026. Assistant messages now carry a `phase` field with
values `commentary` or `final_answer`:

  - `commentary` — intermediate "thinking out loud" text the model emits
    while it's still working (between tool calls, while planning).
  - `final_answer` — the user-facing response the model considers done.

This matters for any agent UI: render commentary in a muted "agent
thinking" panel and the final_answer in the main chat bubble. Without
this distinction, you get the classic UX problem where the agent's
mid-task musings look like the answer.

Reference: OpenAI changelog, Feb 24, 2026.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# === A request that will produce both commentary and a final_answer ===
# Multi-step tasks with tool calls are most likely to surface commentary.

def fake_tool(name, args):
    if name == "get_account_health":
        return '{"health_score": 42, "trend": "declining", "open_p1": 2}'
    if name == "get_recent_activity":
        return '{"last_login": "2026-04-12", "tickets_30d": 7, "champions": 0}'
    return '{"error": "unknown tool"}'


tools = [
    {
        "type": "function", "name": "get_account_health", "strict": True,
        "description": "Get the current health score and trend for an account.",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_recent_activity", "strict": True,
        "description": "Get last login, ticket count, and champion presence.",
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"], "additionalProperties": False,
        },
    },
]


def render(item):
    """Render an output item the way an agent UI would."""
    if item.type == "function_call":
        return f"  [tool] -> {item.name}({item.arguments})"
    if item.type == "message":
        # The phase field is what we're demonstrating here.
        phase = getattr(item, "phase", None)
        text = ""
        for c in item.content:
            if c.type == "output_text":
                text = c.text
        if phase == "commentary":
            return f"  [thinking] {text}"
        elif phase == "final_answer":
            return f"  [ANSWER]   {text}"
        else:
            # Older snapshots may not emit phase
            return f"  [message]  {text}"
    return f"  [{item.type}]"


print("=" * 70)
print("MULTI-STEP REQUEST: agent will reason, call tools, then answer")
print("=" * 70)

import json

response = client.responses.create(
    model="gpt-5.5",
    tools=tools,
    instructions=(
        "You are a CSM assistant. Think step by step about what the user "
        "needs, call tools to gather data, and only commit to a final "
        "recommendation when you have enough evidence."
    ),
    input="Should I escalate ACME-001 to the renewal-risk queue today? Check the data.",
)

# Run the agent loop until done. We'll log each output item and label
# it by phase.
turn = 1
while turn <= 5:
    for item in response.output:
        print(render(item))

    function_calls = [item for item in response.output if item.type == "function_call"]
    if not function_calls:
        break

    print(f"  --- executing {len(function_calls)} tool(s), turn {turn} ---")
    next_input = []
    for fc in function_calls:
        result = fake_tool(fc.name, json.loads(fc.arguments))
        next_input.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": result,
        })

    response = client.responses.create(
        model="gpt-5.5",
        tools=tools,
        previous_response_id=response.id,
        input=next_input,
    )
    turn += 1

# === Counting phases ===

print("\n" + "=" * 70)
print("PHASE BREAKDOWN OF THIS RUN")
print("=" * 70)
phases = {"commentary": 0, "final_answer": 0, "unset": 0, "tool_calls": 0}
all_outputs = response.output
for item in all_outputs:
    if item.type == "function_call":
        phases["tool_calls"] += 1
    elif item.type == "message":
        phase = getattr(item, "phase", None) or "unset"
        phases[phase] = phases.get(phase, 0) + 1
for k, v in phases.items():
    print(f"  {k:<14} {v}")

# === When to use phase ===

print("\n" + "=" * 70)
print("USING phase IN PRODUCTION")
print("=" * 70)
print("""
For agent UIs:
  - Render commentary in a collapsed "agent is working" panel — italic,
    muted color, optionally collapsed by default.
  - Render final_answer prominently — full bubble, persisted in history.

For logging / observability:
  - Persist commentary separately (or drop it) when storing chat history.
    Otherwise users scrolling history see the agent's drafts.

For evals (Exercise 28):
  - Score only the final_answer text. Commentary is intermediate
    state, not the deliverable.

For streaming (Exercise 03):
  - Stream commentary deltas live but mark them as commentary so the UI
    doesn't conflate them with the answer.

Backward compatibility:
  - Older models may not emit phase. Treat missing phase as final_answer
    for safety — you'd rather over-show than hide the answer.
""")

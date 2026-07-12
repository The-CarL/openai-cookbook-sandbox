"""Exercise 35: GPT-5.6 — three tiers, programmatic tool calling, persisted reasoning.

GA: July 9, 2026.  Model IDs: gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna.
The alias gpt-5.6 routes to Sol.

New in GPT-5.6 vs GPT-5.5:
  - Programmatic tool calling: model writes JavaScript that runs in a V8 sandbox
    to orchestrate multiple tool calls without a model round-trip for each call.
  - Persisted reasoning: reasoning.context reuses chain-of-thought across turns.
  - Multi-agent orchestration: beta — model can fan out to parallel subagents.
  - Vision detail settings that preserve original image dimensions.

Pricing per 1M tokens (GA July 9, 2026):
  Sol   $5.00 / $30.00  — flagship; same $/tok as GPT-5.5, stronger on hard tasks
  Terra $2.50 / $15.00  — GPT-5.5-class quality at half the price
  Luna  $1.00 /  $6.00  — fastest / lowest cost

Cache pricing (all three tiers):
  Writes: 1.25× uncached input rate (NEW — prior gens had no write surcharge)
  Reads:  ~10% of uncached input (unchanged)
  Minimum lifetime: 30 minutes; explicit cache breakpoints required.

Reference:
  https://openai.com/index/gpt-5-6/
  https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A customer's RAG pipeline p99 latency jumped from 200ms to 600ms after "
    "switching to a new embedding model. Give a structured 3-step troubleshooting plan."
)

# ── Example 1: Tier comparison ──────────────────────────────────────────────

print("=" * 60)
print("EXAMPLE 1: GPT-5.6 tier comparison (Luna / Terra / Sol)")
print("=" * 60)
print()

TIERS = [
    ("gpt-5.6-luna",  1.00,  6.00, "Fastest / cheapest"),
    ("gpt-5.6-terra", 2.50, 15.00, "Balanced (GPT-5.5 quality)"),
    ("gpt-5.6-sol",   5.00, 30.00, "Flagship"),
]

for model, inp, out, label in TIERS:
    response = client.responses.create(model=model, input=PROMPT)
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    cost = (in_tok / 1_000_000) * inp + (out_tok / 1_000_000) * out
    print(f"[{model}] ({label})")
    print(f"  Tokens: {in_tok} in, {out_tok} out  |  Est. cost: ${cost:.6f}")
    print(f"  Output: {response.output_text[:200]}...")
    print()


# ── Example 2: Programmatic tool calling ────────────────────────────────────
#
# Classic function calling: model decides what to call → you execute → repeat.
# Programmatic tool calling: model writes JavaScript → V8 executes it (no round-trip).
#   - Add {"type": "programmatic_tool_calling"} to tools.
#   - Add "allowed_callers": ["programmatic"] to each eligible function tool.
#   - Model emits "program" items (JS code) + "program_output" items (results).
#   - ZDR-compatible; no extra container costs.
#
# Reference: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling

print("=" * 60)
print("EXAMPLE 2: Programmatic tool calling")
print("=" * 60)
print()
print("Pattern — add programmatic_tool_calling to tools, opt functions in:")
print("""
response = client.responses.create(
    model="gpt-5.6",   # alias for gpt-5.6-sol
    input="Pull inventory for SKUs A1, B2, C3 and flag anything below 10 units.",
    tools=[
        # 1. Enable the programmatic tool calling runtime
        {"type": "programmatic_tool_calling"},

        # 2. Opt each function in via allowed_callers
        {
            "type": "function",
            "name": "get_inventory",
            "description": "Return available units for a SKU.",
            "parameters": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
            },
            "allowed_callers": ["programmatic"],  # let the program call this
        },
    ],
)

# Output items emitted by the model:
#   "message"        — normal text output
#   "program"        — the JavaScript the model wrote (runs in V8 sandbox)
#   "function_call"  — a tool call issued by the program (preserve call_id)
#   "program_output" — the result the V8 runtime returns to the model
for item in response.output:
    print(item.type, getattr(item, "id", ""))
""")

print("Key points:")
print("  - The model writes JS; OpenAI executes it in an isolated V8 runtime.")
print("  - Your function tools are called from within that program — no extra round-trips.")
print("  - Useful when the model needs to call the same tool N times with different args,")
print("    or apply logic to results before deciding the next call.")
print("  - output_schema on function tools gives the program typed results.")
print()


# ── Example 3: Persisted reasoning ──────────────────────────────────────────
#
# GPT-5.6 can reuse its chain-of-thought from prior turns to improve multi-turn
# quality and reduce redundant reasoning tokens.
#
# reasoning.context options:
#   "all_turns"    — model references reasoning from ALL turns in input
#   "current_turn" — ignore prior reasoning (use when context is no longer relevant)
#   "auto"         — model default (same as omitting the field)

print("=" * 60)
print("EXAMPLE 3: Persisted reasoning across turns")
print("=" * 60)
print()

r1 = client.responses.create(
    model="gpt-5.6",
    input="We're designing a RAG pipeline for a legal document corpus. What are the three most important retrieval considerations?",
    reasoning={"effort": "medium", "context": "all_turns"},
)
print(f"Turn 1 ({r1.usage.output_tokens} output tokens):")
print(f"  {r1.output_text[:300]}...")
print()

r2 = client.responses.create(
    model="gpt-5.6",
    input="Good. Now rank those three by implementation difficulty.",
    reasoning={"effort": "medium", "context": "all_turns"},
    previous_response_id=r1.id,
)
print(f"Turn 2 ({r2.usage.output_tokens} output tokens, reasoning reused from turn 1):")
print(f"  {r2.output_text[:300]}...")
print()

print("When to set context='current_turn':")
print("  - The topic or goal shifted significantly mid-conversation.")
print("  - Earlier reasoning is stale (e.g. facts changed, approach was abandoned).")
print("  - Persisted reasoning is adding tokens/latency without quality benefit.")
print()


# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("GPT-5.6 KEY CONCEPTS")
print("=" * 60)
print("""
Model selection in July 2026:
  gpt-5.6-luna   $1/$6/M    — High-volume tasks; fastest tier
  gpt-5.6-terra  $2.50/$15  — Default for quality-sensitive production work
  gpt-5.6-sol    $5/$30     — Hardest reasoning; gpt-5.6 aliases here
  (gpt-5.5 still available and competitive; cheaper if you cache heavily)

Programmatic tool calling:
  tools=[{"type": "programmatic_tool_calling"}, {function + "allowed_callers": ["programmatic"]}]
  The model writes JS → V8 executes → function calls happen without model round-trips.
  Best for: fan-out tool calls, aggregation logic, conditional multi-step tool use.

Persisted reasoning:
  reasoning={"effort": "medium", "context": "all_turns"}
  Reuses prior chain-of-thought; reduces redundant reasoning tokens in multi-turn flows.
  Reset with context="current_turn" when earlier reasoning is no longer relevant.

Caching gotcha (new in 5.6):
  Cache WRITES are billed at 1.25× input rate (unlike 5.5 where writes were free).
  Cache READS remain at ~10% of input.
  30-minute minimum cache lifetime — set explicit breakpoints in your prompt.
""")

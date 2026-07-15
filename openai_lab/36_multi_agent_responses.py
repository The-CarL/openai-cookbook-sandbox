"""Exercise 36: Multi-agent Responses API — parallel subagents in a single request.

Beta feature launched July 9, 2026 with GPT-5.6. When `multi_agent.enabled` is True,
the root model can spawn concurrent subagents and synthesize their results before
returning a final response — all within a single `responses.create()` call.

How it differs from the Agents SDK (ex. 24):
  Agents SDK       — Python-side orchestration; you control the loop
  Multi-agent API  — Model-side orchestration; subagents run inside OpenAI infra
                     → lower latency for fan-out patterns (no Python round-trips)
                     → billing: billed as one request (subagent tokens included)

Key parameter:
  multi_agent = {
      "enabled": True,
      "max_concurrent_subagents": N,   # caps parallelism (default: uncapped)
  }

Only available with GPT-5.6 (Sol, Terra, Luna). Beta as of July 2026.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# --- Example 1: Parallel research synthesis -----------------------------------

print("=" * 60)
print("EXAMPLE 1: Parallel research synthesis")
print("=" * 60)
print()
print("Multi-agent splits this into three independent research streams,")
print("runs them concurrently, then synthesizes a unified answer.")
print()

response = client.responses.create(
    model="gpt-5.6",
    tools=[{"type": "web_search"}],
    multi_agent={
        "enabled": True,
        "max_concurrent_subagents": 3,
    },
    input=(
        "Research these three topics in parallel and synthesize a concise "
        "executive summary:\n"
        "1. Current OpenAI enterprise pricing and tier structure\n"
        "2. Key GPT-5.6 capabilities versus GPT-5.5\n"
        "3. OpenAI Agents SDK latest features (July 2026)\n\n"
        "Combine findings into a 3-paragraph briefing, one paragraph per topic."
    ),
)

print(response.output_text)
print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")


# --- Example 2: Capped concurrency for rate-limit-sensitive tasks ------------

print()
print("=" * 60)
print("EXAMPLE 2: Capped concurrency (max_concurrent_subagents: 2)")
print("=" * 60)
print()
print("Cap concurrency when downstream tools have rate limits or quota.")
print()

response2 = client.responses.create(
    model="gpt-5.6",
    tools=[{"type": "web_search"}],
    multi_agent={
        "enabled": True,
        "max_concurrent_subagents": 2,   # runs two streams at a time
    },
    input=(
        "Analyze four enterprise AI vendors — OpenAI, Google DeepMind, "
        "Anthropic, and Mistral — across these dimensions: (a) pricing, "
        "(b) context window, (c) agentic tooling. "
        "Run each vendor analysis concurrently and return a comparison table."
    ),
)

print(response2.output_text)
print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")


# --- Example 3: Multi-agent without tools (model-only subagents) -------------

print()
print("=" * 60)
print("EXAMPLE 3: Model-only subagents (no tools)")
print("=" * 60)
print()
print("Subagents can also be pure reasoning — no tools required.")
print()

response3 = client.responses.create(
    model="gpt-5.6",
    multi_agent={
        "enabled": True,
        "max_concurrent_subagents": 3,
    },
    input=(
        "A Solutions Engineer is pitching OpenAI's API to a large bank. "
        "Run these three analyses in parallel:\n"
        "1. Security & compliance angle (SOC2, data residency, ZDR)\n"
        "2. Cost model for 10M API calls/month at different tiers\n"
        "3. Integration complexity vs. competitors\n"
        "Then write a one-page executive pitch that weaves all three together."
    ),
)

print(response3.output_text)
print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")


# --- Key concepts summary ------------------------------------------------------

print()
print("=" * 60)
print("MULTI-AGENT RESPONSES API KEY CONCEPTS")
print("=" * 60)
print("""
Parameter:
  multi_agent = {
      "enabled": True,
      "max_concurrent_subagents": N,  # optional; omit for uncapped
  }

How it works:
  1. Root model receives the prompt and tools
  2. Root decides which subtasks to parallelize and spawns subagents
  3. Subagents run concurrently in OpenAI's infrastructure
  4. Root synthesizes subagent results and returns final response
  5. You receive one unified response object (subagent outputs are internal)

Availability:
  GPT-5.6 family only (Sol, Terra, Luna) — gpt-5.4/5.5 do NOT support it.
  Beta as of July 2026.

Billing:
  Billed as a single request. Subagent input/output tokens are included
  in response.usage — there is no per-subagent billing line.

When to use:
  ✓ Work that cleanly separates into independent parallel streams
    (multi-vendor research, multi-topic analysis, parallel code review)
  ✓ When you want model-side orchestration (no Python agent loop)
  ✓ When round-trip latency from the Agents SDK is a concern

When NOT to use:
  ✗ Sequential workflows where step N depends on step N-1 output
  ✗ Tasks requiring Python-side logic between subagents
  ✗ If you need fine-grained control (use Agents SDK instead)

vs. Agents SDK (ex. 24):
  Agents SDK    — Python orchestration, full control, custom guardrails
  Multi-agent   — Model orchestration, lower latency, simpler API surface
""")

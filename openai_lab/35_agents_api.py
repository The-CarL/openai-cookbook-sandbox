"""Exercise 35: Agents API (public beta, Sept 10 2026) — server-managed agent sessions.

The Agents API moves the agent orchestration loop to OpenAI's servers. Unlike:
  - Responses API  — you call responses.create(), you run the loop
  - Agents SDK     — SDK handles the loop client-side; you host it
  - Agents API     — OpenAI handles sessions, context compaction, tool calls,
                     and failure recovery; you pay only for tokens and tools used

Key endpoint:
  client.beta.agents.sessions.create(
      agent={model, instructions, tools, ...},
      environment={type},
      input="...",
      stream=True,
  )

Environment types:
  openai_hosted   — OpenAI provisions a Linux sandbox with Python, Node, curl, etc.
  none            — No execution environment (text/reasoning only)

Billing: tokens + tool calls at standard rates; no additional orchestration fee.

Requires: pip install "openai>=3.8.0"   (beta.agents namespace added in 3.8)
Reference: https://developers.openai.com/api/docs/guides/agents-api
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# --- Example 1: Simple hosted agent (no execution environment) ---------------
print("=" * 60)
print("EXAMPLE 1: Simple agent — reasoning only")
print("=" * 60)
print()
print("Run a single-turn agent session with no hosted sandbox.")
print("Streaming events arrive as JSON; print a summary line per event type.")
print()

event_counts: dict[str, int] = {}

with client.beta.agents.sessions.create(
    agent={
        "model": "gpt-6-astra",
        "instructions": (
            "You are a concise technical advisor. "
            "Answer clearly and briefly — two to four sentences."
        ),
    },
    environment={"type": "none"},
    input=(
        "What are the key differences between OpenAI's Agents API "
        "and the Agents SDK, and when should a developer choose each?"
    ),
    stream=True,
) as events:
    for event in events:
        etype = getattr(event, "type", str(type(event).__name__))
        event_counts[etype] = event_counts.get(etype, 0) + 1

        # Print text deltas inline
        if etype == "agent.message.delta":
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "content"):
                for part in delta.content or []:
                    if hasattr(part, "text"):
                        print(part.text, end="", flush=True)
        elif etype == "agent.session.completed":
            usage = getattr(event, "usage", None)
            if usage:
                print(f"\n\n[tokens] input={usage.input_tokens}, output={usage.output_tokens}")

print()
print("\nEvent type breakdown:")
for etype, count in sorted(event_counts.items()):
    print(f"  {etype:<40} x{count}")


# --- Example 2: Hosted agent with openai_hosted sandbox ----------------------
print()
print("=" * 60)
print("EXAMPLE 2: Hosted agent — Linux sandbox (openai_hosted)")
print("=" * 60)
print()
print("The model can run shell commands, write files, execute Python, etc.")
print("OpenAI provisions and tears down the container automatically.")
print()

event_counts_2: dict[str, int] = {}
final_text = []

with client.beta.agents.sessions.create(
    agent={
        "model": "gpt-6-astra",
        "instructions": (
            "You are a data analysis assistant. "
            "Write and run code to answer the question. Show the actual output."
        ),
    },
    environment={"type": "openai_hosted"},
    input=(
        "Generate a list of 10 random integers between 1 and 100, "
        "compute their mean, median, and standard deviation, "
        "then print a simple ASCII histogram."
    ),
    stream=True,
) as events:
    for event in events:
        etype = getattr(event, "type", str(type(event).__name__))
        event_counts_2[etype] = event_counts_2.get(etype, 0) + 1

        if etype == "agent.message.delta":
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "content"):
                for part in delta.content or []:
                    if hasattr(part, "text"):
                        chunk = part.text
                        final_text.append(chunk)
                        print(chunk, end="", flush=True)
        elif etype == "agent.tool_call":
            tool_name = getattr(event, "tool_name", "?")
            print(f"\n  [tool call] {tool_name}", flush=True)
        elif etype == "agent.tool_result":
            tool_name = getattr(event, "tool_name", "?")
            print(f"  [tool result] {tool_name}", flush=True)
        elif etype == "agent.session.completed":
            usage = getattr(event, "usage", None)
            if usage:
                print(f"\n\n[tokens] input={usage.input_tokens}, output={usage.output_tokens}")

print()
print("\nEvent type breakdown:")
for etype, count in sorted(event_counts_2.items()):
    print(f"  {etype:<40} x{count}")


# --- Summary -----------------------------------------------------------------
print()
print("=" * 60)
print("AGENTS API KEY CONCEPTS")
print("=" * 60)
print("""
When to use the Agents API vs alternatives:

  Responses API     — Full control over the loop; single-model, stateless calls.
                      Best when you want to manage every tool call yourself.

  Agents SDK        — Client-side orchestration with handoffs and guardrails.
                      Best when you want custom agent graphs hosted on your infra.

  Agents API        — OpenAI manages the session, context compaction, and recovery.
                      Best for long-running tasks where you want hosted execution,
                      resilience, and no loop code on your side.

Core call:
  client.beta.agents.sessions.create(
      agent={"model": "...", "instructions": "...", "tools": [...]},
      environment={"type": "openai_hosted"},  # or "none"
      input="...",
      stream=True,
  )

Environment types:
  openai_hosted  — Linux sandbox with Python, Node, curl; suitable for code exec
  none           — Text/reasoning only; no execution environment

Event types (streaming):
  agent.session.started     — Session initialised
  agent.message.delta       — Streaming text chunks
  agent.tool_call           — Model is invoking a tool
  agent.tool_result         — Tool result received
  agent.session.completed   — Run done; includes usage stats

Billing:
  Model tokens (standard Responses API rates) + tool calls at standard rates.
  Container time (openai_hosted): per-minute billing, 5-minute minimum.
  No additional orchestration fee.

SDK requirement: openai>=3.8.0 (beta.agents namespace)
""")

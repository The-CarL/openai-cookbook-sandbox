"""Exercise 35: GPT-5.6 family — Sol, Terra, and Luna.

GPT-5.6 went GA on July 9, 2026. Three durable capability tiers:
  Luna  — fastest, cheapest ($0.20/$1.20 per 1M after Jul 30 cut)
  Terra — balanced ($2.00/$12.00 per 1M after Jul 30 cut)
  Sol   — flagship; best coding and reasoning ($5.00/$30.00 per 1M)

All three share the same 1.05M-token context window (922K input / 128K output)
and support web_search, file_search, code_interpreter, and computer_use.

Fast mode (Sol only): service_tier="fast" gives up to 2.5x the throughput at
2x the price ($10.00/$60.00 per 1M). Falls back to service_tier="priority" for
backward compatibility.
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A SaaS company has a multi-tenant RAG pipeline serving 500 enterprise customers. "
    "They're seeing p99 latency spike from 1.2s to 4.8s after migrating to a new "
    "vector database. Outline a structured debugging plan: what to measure first, "
    "what the likely culprits are, and how to triage them."
)

# Pricing per 1M tokens (verified August 6, 2026 — post-July 30 cuts)
PRICING = {
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00},
    # Fast mode for Sol: 2x the standard rate
    "gpt-5.6-sol-fast": {"input": 10.00, "output": 60.00},
}

MODELS = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]

# --- Example 1: Luna / Terra / Sol standard comparison ---
print("=" * 60)
print("EXAMPLE 1: GPT-5.6 Sol / Terra / Luna comparison")
print("=" * 60)
print()

results = []

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    start = time.time()
    response = client.responses.create(model=model, input=PROMPT)
    elapsed = time.time() - start

    prices = PRICING[model]
    cost = (
        (response.usage.input_tokens / 1_000_000) * prices["input"]
        + (response.usage.output_tokens / 1_000_000) * prices["output"]
    )

    results.append({
        "model": model,
        "elapsed": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": cost,
    })

    print(f"\n{response.output_text}")
    print(f"\n--- {model} stats ---")
    print(f"Latency:  {elapsed:.2f}s")
    print(f"Tokens:   {response.usage.input_tokens} in, {response.usage.output_tokens} out")
    print(f"Est cost: ${cost:.6f}")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<18} {'Latency':>8} {'In tok':>8} {'Out tok':>8} {'Cost':>12}")
print("-" * 60)
for r in results:
    print(
        f"{r['model']:<18} {r['elapsed']:>7.2f}s "
        f"{r['input_tokens']:>8} {r['output_tokens']:>8} ${r['cost']:>10.6f}"
    )

base_sol = next(r for r in results if r["model"] == "gpt-5.6-sol")
print(f"\n--- Relative to Sol ---")
for r in results:
    cost_ratio = r["cost"] / base_sol["cost"] if base_sol["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base_sol["elapsed"] if base_sol["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

# --- Example 2: Fast mode for Sol ---
print()
print("=" * 60)
print("EXAMPLE 2: Fast mode (Sol only) — 2.5× speed at 2× price")
print("=" * 60)
print()
print("Pass service_tier='fast' (or 'priority' for compat) to access fast mode.")
print("Only supported on gpt-5.6-sol. Cost: $10/$60 per 1M tokens.")
print()

start_fast = time.time()
response_fast = client.responses.create(
    model="gpt-5.6-sol",
    input=PROMPT,
    service_tier="fast",
)
elapsed_fast = time.time() - start_fast

fast_cost = (
    (response_fast.usage.input_tokens / 1_000_000) * PRICING["gpt-5.6-sol-fast"]["input"]
    + (response_fast.usage.output_tokens / 1_000_000) * PRICING["gpt-5.6-sol-fast"]["output"]
)

print(f"Fast mode latency:    {elapsed_fast:.2f}s  (standard was {base_sol['elapsed']:.2f}s)")
print(f"Fast mode cost:       ${fast_cost:.6f}  (standard was ${base_sol['cost']:.6f})")

if base_sol["elapsed"] > 0:
    speedup = base_sol["elapsed"] / elapsed_fast
    print(f"Observed speedup:     {speedup:.1f}×  (up to 2.5× per OpenAI)")
if base_sol["cost"] > 0:
    cost_increase = fast_cost / base_sol["cost"]
    print(f"Cost multiplier:      {cost_increase:.1f}×  (2× per OpenAI pricing)")

print()
print("Use fast mode when:")
print("  - You have latency-sensitive real-time applications")
print("  - You need consistent throughput during peak demand")
print("  - The 2× cost premium is worth the speed gain for your use case")

# --- Summary ---
print()
print("=" * 60)
print("GPT-5.6 KEY CONCEPTS")
print("=" * 60)
print("""
GA launch: July 9, 2026. Price cuts: July 30, 2026 (Luna -80%, Terra -20%).

Model IDs:
  gpt-5.6-luna  — fastest, cheapest. $0.20 in / $1.20 out per 1M
  gpt-5.6-terra — balanced.          $2.00 in / $12.00 out per 1M
  gpt-5.6-sol   — flagship.          $5.00 in / $30.00 out per 1M
  gpt-5.6       — alias for Sol

Context window:
  1.05M tokens total (922K input + 128K output) for all three tiers.
  Same ceiling across Luna, Terra, and Sol.

Fast mode (Sol only):
  service_tier="fast" → up to 2.5× throughput at 2× price ($10/$60 per 1M)
  service_tier="priority" still works as a backward-compatible alias.

Prompt caching:
  GPT-5.6 uses explicit cache breakpoints (not automatic).
  30-minute minimum cache lifetime.
  Cache writes billed at 1.25× the standard input rate.
  (Different from 4.1/5.5 which use automatic caching.)

Built-in tools supported by all three tiers:
  web_search, file_search, code_interpreter, computer_use

Picking a tier (August 2026):
  Luna  — high-volume tasks where quality of 5.5 is overkill (routing,
           extraction, summarization). Luna's $0.20/$1.20 makes it
           competitive with gpt-4.1-mini at $0.40/$1.60.
  Terra — the new everyday default. Replaces the 5.4/5.5 mid-tier role.
           $2/$12 undercuts gpt-5.5 ($5/$30) for most production tasks.
  Sol   — hardest reasoning, best coding. Use when quality is non-negotiable.
           Fast mode for latency-critical Sol workloads.
""")

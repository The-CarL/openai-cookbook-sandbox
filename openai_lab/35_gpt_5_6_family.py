"""Exercise 35: GPT-5.6 Sol / Terra / Luna — tier selection and explicit prompt caching.

GA: July 9, 2026. Prices updated July 30, 2026 (Luna -80%, Terra -20% from launch).

Key differences from prior families:
  - New naming: Sol (flagship) / Terra (balanced) / Luna (fast/cheap)
  - 1.05M-token context window on all three tiers
  - Cache WRITES now billed at 1.25x the input rate (reads still 10% of input)
  - Explicit cache breakpoints required — in-memory auto-caching unsupported
  - gpt-5.6 (bare alias) routes to Sol
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Prices per 1M tokens — July 30, 2026 rates
PRICING_56 = {
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20,  "cached_input": 0.02,  "cache_write": 0.25},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cached_input": 0.20,  "cache_write": 2.50},
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00, "cached_input": 0.50,  "cache_write": 6.25},
}

TASK = (
    "A customer is migrating a 10M-row PostgreSQL table to a distributed data warehouse. "
    "They're seeing 3x higher query latency post-migration. Provide a structured "
    "troubleshooting plan with the three most likely root causes and remediation steps."
)


# --- Example 1: Quick tier comparison ---
print("=" * 60)
print("EXAMPLE 1: GPT-5.6 tier comparison (Luna / Terra / Sol)")
print("=" * 60)
print()

results = []
for model in ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]:
    p = PRICING_56[model]
    start = time.time()
    response = client.responses.create(model=model, input=TASK)
    elapsed = time.time() - start

    cost = (
        (response.usage.input_tokens / 1_000_000) * p["input"]
        + (response.usage.output_tokens / 1_000_000) * p["output"]
    )
    results.append({"model": model, "elapsed": elapsed, "cost": cost, "response": response})

    print(f"[{model}]")
    print(f"  Latency: {elapsed:.2f}s | Tokens: {response.usage.input_tokens}in / {response.usage.output_tokens}out | Cost: ${cost:.6f}")
    print(f"  {response.output_text[:200].strip()}...")
    print()

print("Summary:")
sol = next(r for r in results if r["model"] == "gpt-5.6-sol")
for r in results:
    cr = r["cost"] / sol["cost"] if sol["cost"] > 0 else 0
    sr = r["elapsed"] / sol["elapsed"] if sol["elapsed"] > 0 else 0
    print(f"  {r['model']:<18} {cr:>5.1%} cost of Sol, {sr:>5.1%} latency of Sol")


# --- Example 2: Explicit cache breakpoints ---
print()
print("=" * 60)
print("EXAMPLE 2: Explicit cache breakpoints (GPT-5.6 caching model)")
print("=" * 60)
print()

STABLE_SYSTEM = (
    "You are an expert database architect with 20 years of experience in distributed "
    "systems. Your answers are precise, structured, and always actionable. "
    "Reference specific tools and commands when relevant. "
    # In practice this prefix would be thousands of tokens — the stable system prompt
    # that never changes across requests and is worth pinning in the cache.
) * 5  # repeat to simulate a real-sized system prompt

print("Cache billing for GPT-5.6 (unlike prior families):")
print("  Cache writes: 1.25x input rate (you pay to populate the cache)")
print("  Cache reads:  10% of input rate (90% discount on hits)")
print("  Break-even:   A cached prefix becomes net-positive after ~1.39 reuses")
print()

# First request — no cache yet; cache write billed at 1.25x
print("First request (populating cache — write cost applies):")
r1 = client.responses.create(
    model="gpt-5.6-terra",
    instructions=STABLE_SYSTEM,
    input="What are the top three causes of post-migration query latency in PostgreSQL to BigQuery?",
    # prompt_cache_options={"mode": "explicit"} would pin the breakpoint here
    # when the SDK supports it; current SDK uses automatic breakpoints on long prefixes.
)
cached1 = r1.usage.input_tokens_details.cached_tokens if r1.usage.input_tokens_details else 0
p = PRICING_56["gpt-5.6-terra"]
write_cost = ((r1.usage.input_tokens - cached1) / 1_000_000) * p["cache_write"]
read_cost = (cached1 / 1_000_000) * p["cached_input"]
out_cost = (r1.usage.output_tokens / 1_000_000) * p["output"]
print(f"  Input tokens:   {r1.usage.input_tokens} (cached: {cached1})")
print(f"  Cache write:    ${write_cost:.6f}  (1.25x input on uncached tokens)")
print(f"  Cache read:     ${read_cost:.6f}  (10% of input on cached tokens)")
print(f"  Output:         ${out_cost:.6f}")
print(f"  Total:          ${write_cost + read_cost + out_cost:.6f}")
print()

# Second request — same system prompt, different user query; cache should hit
print("Second request (cache read — 90% discount on system prompt tokens):")
r2 = client.responses.create(
    model="gpt-5.6-terra",
    instructions=STABLE_SYSTEM,
    input="How does columnar storage in BigQuery affect join query performance?",
)
cached2 = r2.usage.input_tokens_details.cached_tokens if r2.usage.input_tokens_details else 0
write_cost2 = ((r2.usage.input_tokens - cached2) / 1_000_000) * p["cache_write"]
read_cost2 = (cached2 / 1_000_000) * p["cached_input"]
out_cost2 = (r2.usage.output_tokens / 1_000_000) * p["output"]
print(f"  Input tokens:   {r2.usage.input_tokens} (cached: {cached2})")
print(f"  Cache write:    ${write_cost2:.6f}")
print(f"  Cache read:     ${read_cost2:.6f}")
print(f"  Output:         ${out_cost2:.6f}")
print(f"  Total:          ${write_cost2 + read_cost2 + out_cost2:.6f}")
print(f"  Cache hit rate: {cached2/r2.usage.input_tokens:.0%}" if r2.usage.input_tokens else "")


# --- Summary ---
print()
print("=" * 60)
print("GPT-5.6 KEY FACTS (as of August 15, 2026)")
print("=" * 60)
print("""
Model IDs and July 30, 2026 pricing:
  gpt-5.6-luna   $0.20/$1.20 per 1M tokens  — speed and volume (was $1/$6 at launch)
  gpt-5.6-terra  $2.00/$12.00 per 1M tokens — balanced (was $2.50/$15 at launch)
  gpt-5.6-sol    $5.00/$30.00 per 1M tokens — deepest reasoning (same as gpt-5.5)
  gpt-5.6        bare alias → Sol

Context: 1,050,000 tokens (1.05M) on all three tiers. Max output: 128K tokens.

Caching changes vs prior families:
  Prior (4.1/5.4/5.5): cache writes free; reads billed at 10% of input.
  GPT-5.6+:            cache writes billed at 1.25x input; reads at 10% of input.
  Break-even:          a cached prefix is net-positive after ~1.39 hits.
  Minimum TTL:         30 minutes (same as GPT-5.5 extended caching).
  Breakpoints:         use explicit mode (prompt_cache_options) for deterministic
                       boundaries; in-memory auto-caching is not supported.

When to reach for each tier:
  Luna  — cheap volume: classification, routing, extraction (replaces 4.1-nano/5.4-nano)
  Terra — everyday agentic work, RAG, customer-facing responses
  Sol   — hardest reasoning, coding, research (same price as gpt-5.5)

Ultrafast mode for Sol (limited preview, Aug 2026):
  Up to 14x faster throughput (750 tok/s) via Cerebras infrastructure.
  Available to select API customers; no price change announced yet.
""")

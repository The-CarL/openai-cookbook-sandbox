"""Exercise 35: GPT-5.6 family — Sol/Terra/Luna naming, cache breakpoints, price cuts."""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# GPT-5.6 went GA on July 9, 2026. Key differences from earlier families:
#
# 1. New tier naming — Sol/Terra/Luna replaces base/mini/nano
#      Luna  = cost tier  (~nano equivalent)
#      Terra = balanced   (~mini/base equivalent)
#      Sol   = flagship   (strongest reasoning + coding in the 5.x family)
#
# 2. Explicit cache breakpoints — you mark where cached prefixes end rather
#    than relying on implicit prefix matching.
#
# 3. Cache writes are billable — 1.25× input rate (new; earlier families
#    write to cache for free). Cache reads remain 10% of input (standard rule).
#    Minimum cache lifetime: 30 minutes.
#
# 4. 1.05M-token context window on all three tiers.
#
# Pricing (Sep 28, 2026 — verify at platform.openai.com/docs/pricing):
#   gpt-5.6-luna:  $0.20 input / $1.20 output per 1M tokens
#                  (cut from $1.00/$6.00 on Jul 30, 2026)
#   gpt-5.6-terra: $2.00 input / $12.00 output per 1M tokens
#                  (cut from $2.50/$15.00 on Jul 30, 2026)
#   gpt-5.6-sol:   $4.00 input / $20.00 output per 1M tokens (promotional)
#                  (promotional through Nov 21, 2026; launch price was $5.00/$30.00)
#                  (20% cut applied Aug 21, 2026)

PROMPT = (
    "Explain in two sentences when you'd pick GPT-5.6 Sol over GPT-5.6 Terra "
    "for a production workload."
)

# Pricing for cost tracking in this exercise
PRICING = {
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol":   {"input": 4.00, "output": 20.00},
}

# --- Example 1: Compare all three tiers ---
print("=" * 60)
print("EXAMPLE 1: GPT-5.6 — Luna / Terra / Sol comparison")
print("=" * 60)
print()

results = []
for model in ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]:
    start = time.time()
    response = client.responses.create(model=model, input=PROMPT)
    elapsed = time.time() - start

    p = PRICING[model]
    cost = (response.usage.input_tokens / 1_000_000) * p["input"] + \
           (response.usage.output_tokens / 1_000_000) * p["output"]

    results.append({
        "model": model,
        "elapsed": elapsed,
        "text": response.output_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": cost,
    })
    print(f"[{model}]")
    print(f"  {response.output_text}")
    print(f"  Latency: {elapsed:.2f}s | Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out | Est cost: ${cost:.6f}")
    print()

# --- Example 2: Cache breakpoints (conceptual demonstration) ---
print("=" * 60)
print("EXAMPLE 2: Explicit cache breakpoints (GPT-5.6 feature)")
print("=" * 60)
print()
print("GPT-5.6 requires you to mark where cacheable prefixes end.")
print("Earlier models use implicit prefix matching; 5.6 uses explicit breakpoints.")
print()
print("API call pattern (Python SDK ≥ 1.84.0):")
print("""
  response = client.responses.create(
      model="gpt-5.6-sol",
      input=[
          {
              "role": "system",
              "content": [
                  {"type": "text", "text": "<long system prompt here>"},
                  # Mark the cache breakpoint after your static prefix:
                  {"type": "cache_control", "cache_type": "ephemeral"},
              ],
          },
          {"role": "user", "content": "User message here"},
      ],
  )
""")
print("Cost model for cache breakpoints:")
print("  Write:  1.25× input rate (billed on the first request that sets the breakpoint)")
print("  Read:   0.10× input rate (billed on subsequent requests that hit the cache)")
print("  Minimum cache lifetime: 30 minutes")
print()
print("Important: if you DON'T set a breakpoint, GPT-5.6 will NOT automatically")
print("cache your prefix — you must opt in explicitly (unlike GPT-5.4/5.5).")

# --- Example 3: Run a real cached call ---
print()
print("=" * 60)
print("EXAMPLE 3: Two calls with the same prefix — verify cache hit on second")
print("=" * 60)
print()

STATIC_SYSTEM = (
    "You are a helpful API documentation assistant. "
    "Always respond in a single concise sentence. "
    * 30  # ~200 tokens — enough to be worth caching
)

input_with_breakpoint = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": STATIC_SYSTEM},
            {"type": "cache_control", "cache_type": "ephemeral"},
        ],
    },
    {"role": "user", "content": "What is the Responses API?"},
]

print("First call (cache miss — breakpoint written at 1.25× input rate):")
r_first = client.responses.create(
    model="gpt-5.6-terra",
    input=input_with_breakpoint,
)
cached_first = r_first.usage.input_tokens_details.cached_tokens if r_first.usage.input_tokens_details else 0
print(f"  Tokens: {r_first.usage.input_tokens} in ({cached_first} cached), {r_first.usage.output_tokens} out")
print(f"  Response: {r_first.output_text}")

print()
print("Second call (cache hit expected — reads at 0.10× input rate):")
input_with_breakpoint[1]["content"] = "What is context compaction?"
r_second = client.responses.create(
    model="gpt-5.6-terra",
    input=input_with_breakpoint,
)
cached_second = r_second.usage.input_tokens_details.cached_tokens if r_second.usage.input_tokens_details else 0
print(f"  Tokens: {r_second.usage.input_tokens} in ({cached_second} cached), {r_second.usage.output_tokens} out")
print(f"  Response: {r_second.output_text}")

if cached_second > 0:
    print(f"\n  Cache hit confirmed: {cached_second} tokens read from cache.")
else:
    print("\n  No cache hit on second call (can happen if breakpoint didn't persist or min lifetime not met).")

# --- Summary ---
print()
print("=" * 60)
print("GPT-5.6 QUICK REFERENCE")
print("=" * 60)
print("""
Model IDs:
  gpt-5.6-luna    — cost tier    ($0.20 / $1.20 per 1M — after Jul 30 cut)
  gpt-5.6-terra   — balanced     ($2.00 / $12.00 per 1M — after Jul 30 cut)
  gpt-5.6-sol     — flagship     ($4.00 / $20.00 per 1M — promotional to Nov 21)

Context:  1.05M tokens on all tiers

Naming history:
  GPT-5.6 introduces Sol/Terra/Luna — OpenAI's new canonical tier names.
  Older models still use nano/mini/base naming; new families from 5.6 onward use S/T/L.

Cache breakpoints (GPT-5.6-specific):
  - Required: you MUST set {"type": "cache_control", "cache_type": "ephemeral"}
    at the end of your static prefix — implicit caching does NOT apply.
  - Write cost: 1.25× input rate (first call setting the breakpoint)
  - Read cost:  0.10× input rate (subsequent cache hits)
  - Min lifetime: 30 minutes

When to pick each tier:
  Luna:   Bulk classification, routing, cheap extraction where quality > 4.1-nano.
  Terra:  The new default for production agentic workloads (replaces 5.4-mini).
  Sol:    Hardest reasoning/coding tasks in the 5.x family; strongest cybersec eval scores.

Price-cut timeline (Luna/Terra — Sol promotional):
  July 9, 2026:   GA launch — Luna $1/$6, Terra $2.50/$15, Sol $5/$30
  July 30, 2026:  Luna −80% → $0.20/$1.20; Terra −20% → $2.00/$12.00; Sol unchanged
  Aug 21, 2026:   Sol −20% → $4.00/$20.00 (promotional through Nov 21, 2026)
""")

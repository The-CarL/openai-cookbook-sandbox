"""Exercise 4: Compare GPT-4.1, GPT-5.4, GPT-5.5, and GPT-5.6 model families."""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A customer says: 'We're seeing 3x higher latency on our RAG pipeline since "
    "migrating to the new embedding model. Our p99 went from 200ms to 600ms. "
    "What should we investigate?' "
    "Give a structured troubleshooting plan."
)

MODELS = [
    # GPT-4.1 family — cost-effective workhorse (1M context, no native reasoning)
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    # GPT-5.4 family — March 2026 flagship (native reasoning, computer use, image gen)
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
    # GPT-5.5 — April 23, 2026 flagship. More token-efficient than 5.4 on most tasks
    # but ~2x per-token price. Often cheaper end-to-end.
    "gpt-5.5",
    # GPT-5.6 family — GA July 9, 2026. Three tiers: Luna (fast/cheap), Terra (balanced),
    # Sol (frontier). New cache model: explicit breakpoints, writes at 1.25x input,
    # reads at 10%. Luna/Terra prices cut 80%/20% on July 30, 2026.
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
]

# Pricing per 1M tokens (verified August 7, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
# GPT-5.6 cache writes are billed at 1.25x the input rate (Sol $6.25, Terra $2.50,
# Luna $0.25); cache reads remain ~10% of input. Explicit breakpoints only.
PRICING = {
    "gpt-4.1-nano":   {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-5.4-nano":   {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":   {"input": 0.75,  "output": 4.50},
    "gpt-5.4":        {"input": 2.50,  "output": 15.00},
    "gpt-5.5":        {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    # GPT-5.6 prices post July-30 cuts; Sol unchanged; Terra -20%; Luna -80%
    "gpt-5.6-luna":   {"input": 0.20,  "output": 1.20},
    "gpt-5.6-terra":  {"input": 2.00,  "output": 12.00},
    "gpt-5.6-sol":    {"input": 5.00,  "output": 30.00},
}

results = []

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    start = time.time()
    response = client.responses.create(model=model, input=PROMPT)
    elapsed = time.time() - start

    cost_input = (response.usage.input_tokens / 1_000_000) * PRICING[model]["input"]
    cost_output = (response.usage.output_tokens / 1_000_000) * PRICING[model]["output"]
    cost_total = cost_input + cost_output

    results.append({
        "model": model,
        "elapsed": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": cost_total,
        "text": response.output_text,
    })

    print(f"\n{response.output_text}")
    print(f"\n--- {model} stats ---")
    print(f"Latency:  {elapsed:.2f}s")
    print(f"Tokens:   {response.usage.input_tokens} in, {response.usage.output_tokens} out")
    print(f"Est cost: ${cost_total:.6f}")

# Summary comparison
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<18} {'Latency':>8} {'In tok':>8} {'Out tok':>8} {'Cost':>12}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<18} {r['elapsed']:>7.2f}s {r['input_tokens']:>8} {r['output_tokens']:>8} ${r['cost']:>10.6f}")

print("\n--- Relative to gpt-5.6-sol (current frontier) ---")
base = results[-1]  # gpt-5.6-sol
for r in results:
    cost_ratio = r["cost"] / base["cost"] if base["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

print("\n--- Picking a model in August 2026 ---")
print("GPT-4.1 family (1M context, no native reasoning):")
print("  nano:  Classification, routing, simple extraction at the lowest price.")
print("  mini:  Sweet spot for high-volume production where 5.x is overkill.")
print("  4.1:   When you need 1M context but not reasoning.")
print()
print("GPT-5.4 family (native reasoning, computer use, image gen):")
print("  nano:  Budget reasoning. Better than 4.1-nano on hard tasks.")
print("  mini:  Strong default for agentic workloads. Cheaper than 5.6-luna on output.")
print("  5.4:   Cheaper per-token than 5.5/5.6-sol — keep for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  Still strong. Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session.")
print()
print("GPT-5.6 family (GA July 9, 2026; Luna/Terra prices cut July 30):")
print("  luna:  $0.20/$1.20 — fastest and cheapest 5.6 tier; good for routing/drafts.")
print("  terra: $2.00/$12.00 — balanced; similar price to 5.4 but newer generation.")
print("  sol:   $5.00/$30.00 — frontier; best coding and complex reasoning.")
print("  Caching gotcha: cache WRITES are billed at 1.25x input (not free).")
print("  Cache reads remain ~10%. Use explicit prompt_cache_breakpoint to control splits.")
print("  Fast mode (sol only): $10/$60 per 1M, up to 2.5x faster than standard.")

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
    # GPT-5.4 family — March 2026 (native reasoning, computer use, image gen)
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
    # GPT-5.5 — April 23, 2026. Often cheaper end-to-end than 5.4 despite higher
    # per-token price. Keep for workloads already tuned to 5.5.
    "gpt-5.5",
    # GPT-5.6 family — GA July 9, 2026. Stronger agent performance + explicit
    # cache breakpoints (see ex. 35). Prices after July 30 cuts: Luna -80%, Terra -20%.
    # gpt-5.6 is an alias for gpt-5.6-sol.
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
]

# Pricing per 1M tokens (verified August 16, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
# GPT-5.6 cache WRITES cost 1.25x the input rate (unlike 5.5 where writes were free).
# GPT-5.6 long-context: Sol 2x/1.5x at $10/$45, Terra $4/$18, Luna $0.40/$1.80.
PRICING = {
    "gpt-4.1-nano":   {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini":   {"input": 0.40, "output": 1.60},
    "gpt-4.1":        {"input": 2.00, "output": 8.00},
    "gpt-5.4-nano":   {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini":   {"input": 0.75, "output": 4.50},
    "gpt-5.4":        {"input": 2.50, "output": 15.00},
    "gpt-5.5":        {"input": 5.00, "output": 30.00},  # standard (<=272K input)
    # GPT-5.6 after July 30, 2026 price cuts
    "gpt-5.6-luna":   {"input": 0.20, "output": 1.20},   # was $1.00/$6.00 pre-cut
    "gpt-5.6-terra":  {"input": 2.00, "output": 12.00},  # was $2.50/$15.00 pre-cut
    "gpt-5.6-sol":    {"input": 5.00, "output": 30.00},  # unchanged; alias: gpt-5.6
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

print("\n--- Relative to gpt-5.6-sol (current flagship) ---")
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
print("  mini:  The default for most new agentic workloads.")
print("  5.4:   Still strong; cheaper per-token than 5.5/5.6 — keep for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  More token-efficient than 5.4. Cache writes are free (unlike 5.6).")
print("  Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session, not just the overage.")
print()
print("GPT-5.6 family (GA July 9, 2026 — prices after July 30, 2026 cuts):")
print("  luna:  $0.20/$1.20 per 1M (was $1/$6 pre-cut). High-volume budget tier.")
print("  terra: $2.00/$12.00 per 1M (was $2.50/$15 pre-cut). Balanced mid-tier.")
print("  sol:   $5.00/$30.00 per 1M (unchanged). Current flagship; alias: gpt-5.6.")
print("  All 5.6 models: explicit cache breakpoints, 30-min min TTL, writes at 1.25x.")
print("  See exercise 35 for cache-breakpoint billing and prompt_cache_options API.")

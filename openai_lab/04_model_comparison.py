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
    # GPT-5.6 family — July 9, 2026 GA. 1.05M context window.
    # gpt-5.6 aliases to Sol. Cache writes billed at 1.25x input (see ex. 25/26).
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
]

# Pricing per 1M tokens (verified July 30, 2026)
# Long-context surcharge (>272K input tokens) applies to GPT-5.5 and GPT-5.6:
#   5.5:         $10.00/$45.00 (2x input / 1.5x output) for the ENTIRE session
#   5.6-sol:     $10.00/$45.00
#   5.6-terra:   $5.00/$22.50
#   5.6-luna:    $2.00/$9.00
PRICING = {
    "gpt-4.1-nano":   {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-5.4-nano":   {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":   {"input": 0.75,  "output": 4.50},
    "gpt-5.4":        {"input": 2.50,  "output": 15.00},
    "gpt-5.5":        {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    "gpt-5.6-sol":    {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    "gpt-5.6-terra":  {"input": 2.50,  "output": 15.00},
    "gpt-5.6-luna":   {"input": 1.00,  "output": 6.00},
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
base = results[-1]  # gpt-5.6-luna; sol is results[-3]
for r in results:
    cost_ratio = r["cost"] / base["cost"] if base["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

print("\n--- Picking a model in July 2026 ---")
print("GPT-4.1 family (1M context, no native reasoning):")
print("  nano:  Classification, routing, simple extraction at the lowest price.")
print("  mini:  Sweet spot for high-volume production where 5.x is overkill.")
print("  4.1:   When you need 1M context but not reasoning.")
print()
print("GPT-5.4 family (native reasoning, computer use, image gen):")
print("  nano:  Budget reasoning. Better than 4.1-nano on hard tasks.")
print("  mini:  Still a solid default for agentic workloads; lower cost than 5.6-terra.")
print("  5.4:   Cost-sensitive flows where 5.6 would be overkill.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  Per-token cost same as 5.6-sol but 5.6-sol is newer and often stronger.")
print("  Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session, not just the overage.")
print()
print("GPT-5.6 family (July 9, 2026 GA — 1.05M context, 128K max output):")
print("  sol:   Flagship. Best for hard coding, research, computer use, tool-heavy work.")
print("         Alias: gpt-5.6 → gpt-5.6-sol. Same price as gpt-5.5 per token.")
print("  terra: Balanced quality/cost. Same per-token price as gpt-5.4.")
print("  luna:  Cost-focused. Good for high-volume tasks that don't need Sol-level quality.")
print("  Cache: writes billed at 1.25x input rate; reads at 0.1x (same 10% rule).")
print("  Long-context (>272K): sol $10/$45, terra $5/$22.50, luna $2/$9 per 1M.")

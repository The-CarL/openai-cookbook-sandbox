"""Exercise 4: Compare GPT-4.1, GPT-5.4, and GPT-5.5 model families."""

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
    # GPT-5.6 family — GA July 9, 2026. Sol=flagship, Terra=balanced, Luna=budget.
    # New: cache writes billed at 1.25× input; 1.05M context; better reasoning+coding.
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
]

# Pricing per 1M tokens (verified July 16, 2026)
# GPT-5.5/5.6 long-context: sessions >272K input tokens are billed at 2x input
# and 1.5x output for the ENTIRE session.
# GPT-5.6 ONLY: cache writes cost 1.25× input (prior models: writes were free).
PRICING = {
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1":       {"input": 2.00, "output": 8.00},
    "gpt-5.4-nano":  {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75, "output": 4.50},
    "gpt-5.4":       {"input": 2.50, "output": 15.00},
    "gpt-5.5":       {"input": 5.00, "output": 30.00},  # standard (<=272K input)
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00},  # Sol: same price as 5.5, better quality
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00},  # Terra: same price as 5.4
    "gpt-5.6-luna":  {"input": 1.00, "output": 6.00},   # Luna: new budget tier
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

print("\n--- Relative to gpt-5.5 (current flagship) ---")
base = results[-1]  # gpt-5.5
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
print("  mini:  The default for most new agentic workloads.")
print("  5.4:   Still strong; cheaper per-token than 5.5 — keep for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  More token-efficient than 5.4. Cache writes are free (unlike 5.6).")
print("  Still valid; reach for 5.6-sol when you want the current frontier.")
print()
print("GPT-5.6 family (GA July 9, 2026) — NEW:")
print("  Sol ($5/$30):   New flagship. Same price as 5.5, stronger on reasoning/coding/cybersec.")
print("                  'gpt-5.6' alias → Sol. Long-context (>272K): 2×/1.5× for full session.")
print("  Terra ($2.50/$15): Balanced tier. Same price as 5.4 with 5.6 quality gains.")
print("  Luna ($1/$6):   Budget tier. New name convention (was 'nano'); cheapest 5.6.")
print()
print("  CACHING CHANGE in 5.6: writes now billed at 1.25× input rate.")
print("  Cache reads still 90% discount. Break-even at ~1.25 read hits per write.")

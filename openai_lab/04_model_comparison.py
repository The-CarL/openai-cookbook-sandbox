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
    # GPT-5.5 — April 23, 2026. Token-efficient; strong default until 5.6 is needed.
    "gpt-5.5",
    # GPT-5.6 family (GA July 9, 2026). New Sol/Terra/Luna tier naming.
    # Cache writes billed at 1.25x uncached input (new for 5.6+).
    # Sol promo pricing $4/$20 through Nov 21, 2026 (standard $5/$30).
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
]

# Pricing per 1M tokens (verified Sept 12, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
# GPT-5.6 Sol promotional rate ($4/$20) valid through Nov 21, 2026 (standard $5/$30).
PRICING = {
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1":       {"input": 2.00, "output": 8.00},
    "gpt-5.4-nano":  {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75, "output": 4.50},
    "gpt-5.4":       {"input": 2.50, "output": 15.00},
    "gpt-5.5":       {"input": 5.00, "output": 30.00},  # standard (<=272K input)
    # GPT-5.6 post-cut prices (Luna -80% and Terra -20% on Jul 30; Sol promo from Aug 21)
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol":   {"input": 4.00, "output": 20.00},
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

print("\n--- Relative to gpt-5.6-sol (current 5.6 flagship) ---")
base = next(r for r in results if r["model"] == "gpt-5.6-sol")
for r in results:
    cost_ratio = r["cost"] / base["cost"] if base["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

print("\n--- Picking a model in Sept 2026 ---")
print("GPT-4.1 family (1M context, no native reasoning):")
print("  nano:  Classification, routing, simple extraction at the lowest price.")
print("  mini:  Sweet spot for high-volume production where 5.x is overkill.")
print("  4.1:   When you need 1M context but not reasoning.")
print()
print("GPT-5.4 family (native reasoning, computer use, image gen):")
print("  nano:  Budget reasoning. Better than 4.1-nano on hard tasks.")
print("  mini:  Default for agentic workloads where 5.6-luna is overkill.")
print("  5.4:   Still strong; cheaper than 5.5 per-token for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  Token-efficient relative to 5.4. Strong default for high-quality flows.")
print("  Long-context gotcha: sessions >272K input are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session.")
print()
print("GPT-5.6 family (GA July 9, 2026 — Sol/Terra/Luna tier naming):")
print("  luna:  Fast/affordable. $0.20/$1.20/M (cut 80% from launch Jul 30).")
print("  terra: Balanced everyday tier. $2.00/$12.00/M (cut 20% from launch Jul 30).")
print("  sol:   5.6 flagship. $4/$20/M promo through Nov 21, 2026 (std $5/$30).")
print("         Context: 1.05M tokens (922K input / 128K output max).")
print()
print("GPT-5.6 caching (new for 5.6+):")
print("  - Explicit cache breakpoints + 30-min minimum cache lifetime.")
print("  - Cache WRITES billed at 1.25x uncached input rate.")
print("  - Cache reads: still 10% of uncached (90% discount unchanged).")

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
    # but ~2x per-token price. Often cheaper end-to-end. Use this as the new default
    # for any task where 5.4-mini is too weak.
    "gpt-5.5",
]

# Pricing per 1M tokens (verified April 27, 2026)
# NOTE: gpt-5.5 long-context surcharge — prompts >272K input tokens are billed
# at 2x input ($10.00/M) and 1.5x output ($45.00/M) for the full session.
PRICING = {
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1":       {"input": 2.00, "output": 8.00},
    "gpt-5.4-nano":  {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75, "output": 4.50},
    "gpt-5.4":       {"input": 2.50, "output": 15.00},
    "gpt-5.5":       {"input": 5.00, "output": 30.00},
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

print("\n--- Picking a model in April 2026 ---")
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
print("  More token-efficient than 5.4 for most tasks, so often cheaper end-to-end")
print("  even at 2x the per-token price. Default choice for new high-quality flows.")
print("  Note: shell tool docs and most new examples use gpt-5.5.")

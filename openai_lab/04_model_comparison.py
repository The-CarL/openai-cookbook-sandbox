"""Exercise 4: Compare GPT-4.1, GPT-5.4, GPT-5.5, and GPT-6 Astra model families."""

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
    # GPT-6 Astra — September 3, 2026 flagship. SOTA computer use, SWE, cybersecurity.
    # 1,050,000-token context, 128K max output. Fast mode available at 2x speed/price.
    "gpt-6-astra",
]

# Pricing per 1M tokens (verified September 9, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
PRICING = {
    "gpt-4.1-nano":  {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40,  "output": 1.60},
    "gpt-4.1":       {"input": 2.00,  "output": 8.00},
    "gpt-5.4-nano":  {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75,  "output": 4.50},
    "gpt-5.4":       {"input": 2.50,  "output": 15.00},
    "gpt-5.5":       {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    "gpt-6-astra":   {"input": 10.00, "output": 50.00},  # Sep 3 2026; Fast mode 2x
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

print("\n--- Relative to gpt-6-astra (current flagship) ---")
base = next(r for r in results if r["model"] == "gpt-6-astra")
for r in results:
    cost_ratio = r["cost"] / base["cost"] if base["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

print("\n--- Picking a model in September 2026 ---")
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
print("  even at 2x the per-token price. Still a strong choice for high-quality flows.")
print("  Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session, not just the overage.")
print()
print("GPT-6 Astra (September 3, 2026 — current flagship):")
print("  $10.00/$50.00 per 1M. 1,050,000-token context, 128K max output.")
print("  State-of-the-art on computer use, browsing, SWE, cybersecurity, science.")
print("  Available on Responses, Chat Completions, Realtime, and Batch APIs.")
print("  Fast mode: 2x speed at 2x price. Batch: 50% off standard rate.")
print("  Astra can pause mid-task for human review when it detects instruction ambiguity.")

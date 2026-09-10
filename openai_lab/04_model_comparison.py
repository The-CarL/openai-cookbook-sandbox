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
    # GPT-5.5 — April 23, 2026. More token-efficient than 5.4 on most tasks.
    "gpt-5.5",
    # GPT-5.6 — GA July 9, 2026. Three tiers: Luna (budget), Terra (mid), Sol (flagship).
    # Introduces explicit cache breakpoints with 30-min minimum lifetime;
    # cache writes billed at 1.25x input rate. Stronger reasoning, coding, cybersecurity.
    # Sol is at promotional pricing ($4/$20) through Nov 21, 2026; list price is $5/$30.
    # Terra and Luna had 20% and 80% price cuts on July 30, 2026.
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
]

# Pricing per 1M tokens (verified September 10, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
# GPT-5.6-Sol promotional: $4/$20 through Nov 21, 2026; list price is $5/$30.
PRICING = {
    "gpt-4.1-nano":  {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40,  "output": 1.60},
    "gpt-4.1":       {"input": 2.00,  "output": 8.00},
    "gpt-5.4-nano":  {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":  {"input": 0.75,  "output": 4.50},
    "gpt-5.4":       {"input": 2.50,  "output": 15.00},
    "gpt-5.5":       {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    "gpt-5.6-luna":  {"input": 0.20,  "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00,  "output": 12.00},
    "gpt-5.6-sol":   {"input": 4.00,  "output": 20.00},  # promo through Nov 21; list $5/$30
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

print("\n--- Picking a model in September 2026 ---")
print("GPT-4.1 family (1M context, no native reasoning):")
print("  nano:  Classification, routing, simple extraction at the lowest price.")
print("  mini:  Sweet spot for high-volume production where 5.x is overkill.")
print("  4.1:   When you need 1M context but not reasoning.")
print()
print("GPT-5.4 family (native reasoning, computer use, image gen):")
print("  nano:  Budget reasoning. Better than 4.1-nano on hard tasks.")
print("  mini:  Solid for agentic workloads where 5.6 is overkill.")
print("  5.4:   Cheaper per-token than 5.6 Sol — keep for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  Token-efficient flagship before 5.6. Still valid but superseded by 5.6 Sol.")
print("  Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session, not just the overage.")
print()
print("GPT-5.6 (GA July 9, 2026) — current flagship family:")
print("  luna:  Budget tier. $0.20/$1.20/M. Best quality-per-dollar for simple tasks.")
print("  terra: Mid-tier. $2/$12/M. Replaces 5.5 for cost-sensitive agentic work.")
print("  sol:   Flagship. $4/$20/M promo (through Nov 21, 2026; list $5/$30).")
print("         54% more token-efficient on coding tasks than 5.5.")
print("  Cache note: explicit cache breakpoints; writes at 1.25x input rate, reads at 10%.")

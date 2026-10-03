"""Exercise 4: Compare GPT-4.1, GPT-5.4, GPT-5.5, GPT-5.6, and GPT-6 model families."""

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
    # GPT-5.5 — April 23, 2026. Token-efficient; good fallback if 5.6/6 unavailable.
    "gpt-5.5",
    # GPT-5.6 family — GA July 9, 2026 (1.05M ctx). Cache writes billed at 1.25x input.
    # Sol promo ($4/$20) active through at least Nov 21, 2026 (standard: $5/$30).
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
    # GPT-6 family — September 22, 2026. Text+image input. 50% cheaper than 5.6.
    # Cache writes billed at 1.25x input rate (same as 5.6+). 1.05M context.
    "gpt-6-luna", "gpt-6-sol",
    # GPT-6.1 Sol — September 29, 2026 (DevDay). Near-Astra performance for coding
    # and computer use at 1/5 of Astra's price. Same I/O pricing as gpt-6-sol but
    # cached input cut to $0.10/M (vs $0.20/M for gpt-6-sol). 1.05M context.
    "gpt-6.1-sol",
    # GPT-6 Astra — September 3, 2026. Hardest reasoning/coding/computer use/research.
    # 1M context; Fast mode at 2x standard rates; cache writes at 1.25x input.
    "gpt-6-astra",
]

# Pricing per 1M tokens (verified Oct 3, 2026)
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
# GPT-5.6 Sol: promotional rate through at least Nov 21, 2026 (standard: $5/$30).
# GPT-6 Astra: >272K input uses 2x input + 1.5x output for the full session.
# Astra Fast mode: $20/$100 per 1M; comparison below uses standard rates.
# GPT-6.1 Sol: cached input $0.10/M (50% below gpt-6-sol's $0.20/M).
PRICING = {
    "gpt-4.1-nano":   {"input": 0.10,  "output": 0.40},
    "gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-5.4-nano":   {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":   {"input": 0.75,  "output": 4.50},
    "gpt-5.4":        {"input": 2.50,  "output": 15.00},
    "gpt-5.5":        {"input": 5.00,  "output": 30.00},  # standard (<=272K input)
    "gpt-5.6-luna":   {"input": 0.20,  "output": 1.20},
    "gpt-5.6-terra":  {"input": 2.00,  "output": 12.00},
    "gpt-5.6-sol":    {"input": 4.00,  "output": 20.00},  # promo; see note above
    "gpt-6-luna":     {"input": 0.10,  "output": 0.50},
    "gpt-6-sol":      {"input": 2.00,  "output": 10.00},
    "gpt-6.1-sol":    {"input": 2.00,  "output": 10.00},  # cached: $0.10/M (see ex. 18)
    "gpt-6-astra":    {"input": 10.00, "output": 50.00},  # standard, <=272K input
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

BASE_MODEL = "gpt-6-sol"
print(f"\n--- Relative to {BASE_MODEL} (comparison baseline) ---")
base = next(r for r in results if r["model"] == BASE_MODEL)
for r in results:
    cost_ratio = r["cost"] / base["cost"] if base["cost"] > 0 else 0
    speed_ratio = r["elapsed"] / base["elapsed"] if base["elapsed"] > 0 else 0
    print(f"{r['model']:<18} {cost_ratio:>5.1%} the cost, {speed_ratio:>5.1%} the latency")

print("\n--- Picking a model in Oct 2026 ---")
print("GPT-4.1 family (1M context, no native reasoning):")
print("  nano:  Classification, routing, simple extraction at the lowest price.")
print("  mini:  Sweet spot for high-volume production where 5.x is overkill.")
print("  4.1:   When you need 1M context but not reasoning.")
print()
print("GPT-5.4 family (native reasoning, computer use, image gen):")
print("  nano:  Budget reasoning. Better than 4.1-nano on hard tasks.")
print("  mini:  A solid choice for legacy agentic workloads needing computer use.")
print("  5.4:   Still strong; cheaper per-token than 5.5 — keep for cost-sensitive flows.")
print()
print("GPT-5.5 (April 23, 2026):")
print("  Token-efficient. Good fallback when 5.6/6 are unavailable or long-context needed.")
print("  Long-context gotcha: sessions >272K input tokens are billed at")
print("  $10.00/$45.00 per 1M (2x/1.5x) for the full session, not just the overage.")
print()
print("GPT-5.6 family (GA July 9, 2026 — 1.05M context):")
print("  luna:  $0.20/$1.20. Budget 5.6 tier. Replaces 5.5 for cost-sensitive flows.")
print("  terra: $2.00/$12.00. Balanced; stronger coding/cybersecurity than 5.5.")
print("  sol:   $4.00/$20.00 (promo through Nov 2026). Hardest tasks in the 5.x line.")
print("  Caching: explicit breakpoints; writes 1.25x input, reads ~10%, 30-min min lifetime.")
print("  See Exercise 35 for the focused GPT-5.6 walkthrough and price-cut timeline.")
print()
print("GPT-6 family (September 22, 2026 — text+image input):")
print("  luna: $0.10/$0.50. Cheapest capable model; 50% below 5.6 Luna.")
print("  sol:  $2.00/$10.00. Default for new high-quality flows; 50% below 5.6 Sol.")
print("  Cache write billing (1.25x input) applies, same as 5.6+.")

print()
print("GPT-6.1 Sol (September 29, 2026 — DevDay flagship):")
print("  Near-Astra performance for coding and computer use at 1/5 of Astra's price.")
print("  $2/$10 per 1M (same I/O as gpt-6-sol). Cached input: $0.10/M (50% below gpt-6-sol).")
print("  1.05M context, 128K max output. Default upgrade path from gpt-6-sol.")
print()
print("GPT-6 Astra (September 3, 2026 — premium frontier model):")
print("  Hardest reasoning, coding, computer use, and research; 1M context.")
print("  $10/$50 per 1M standard; Fast mode $20/$100. Cached input: $1/M.")
print("  Long-context: >272K input means 2x input + 1.5x output for the full session.")
print("  Cache writes: 1.25x input ($12.50/M). Estimates above use standard rates.")

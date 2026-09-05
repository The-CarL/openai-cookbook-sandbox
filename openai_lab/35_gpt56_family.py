"""Exercise 35: GPT-5.6 family — Sol, Terra, and Luna.

GPT-5.6 became broadly available on July 30, 2026. Three tiers:
  gpt-5.6-sol   — $4.00/$20.00/M  Strongest reasoning + coding; new default flagship
  gpt-5.6-terra — $2.00/$12.00/M  Balanced mid-tier (Terra price-dropped July 30)
  gpt-5.6-luna  — $0.20/$1.20/M   Budget tier; 80% cheaper than original (Luna drop July 30)

Caching (different from 4.1/5.4/5.5):
  - Explicit cache breakpoints in the prompt (you mark them via the API)
  - 30-minute minimum cache lifetime
  - Cache WRITES billed at 1.25x input rate
  - Cache READS discounted (~10% of input rate)
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A production service needs to choose between three approaches for handling "
    "a large-scale data pipeline: (1) streaming with Apache Kafka, "
    "(2) batch with Apache Spark, (3) real-time with Apache Flink. "
    "Evaluate each option across latency, throughput, operational complexity, "
    "and cost at 10TB/day scale. Give a concrete recommendation."
)

MODELS = [
    ("gpt-5.6-luna",  0.20,  1.20),  # ($input/M, $output/M)
    ("gpt-5.6-terra", 2.00, 12.00),
    ("gpt-5.6-sol",   4.00, 20.00),
]

# Pricing per 1M tokens (verified Sept 5, 2026)
# Note: Sol price cut Aug 2026 (was $5.00/$30.00 at limited preview launch)
PRICING = {
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol":   {"input": 4.00, "output": 20.00},
}

print("=" * 60)
print("GPT-5.6 FAMILY COMPARISON: Sol / Terra / Luna")
print("=" * 60)

results = []

for model, price_in, price_out in MODELS:
    print(f"\n{'─'*60}")
    print(f"MODEL: {model}  (${price_in}/${price_out} per 1M in/out)")
    print(f"{'─'*60}")

    start = time.time()
    response = client.responses.create(model=model, input=PROMPT)
    elapsed = time.time() - start

    cost_in = (response.usage.input_tokens / 1_000_000) * price_in
    cost_out = (response.usage.output_tokens / 1_000_000) * price_out
    total = cost_in + cost_out

    results.append({
        "model": model,
        "elapsed": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": total,
        "text": response.output_text,
    })

    print(f"\n{response.output_text[:600]}...")
    print(f"\nLatency: {elapsed:.2f}s | In: {response.usage.input_tokens} | "
          f"Out: {response.usage.output_tokens} | Cost: ${total:.6f}")

# Summary
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<18} {'Latency':>9} {'In tok':>8} {'Out tok':>8} {'Cost':>12}")
print("─" * 60)
for r in results:
    print(f"{r['model']:<18} {r['elapsed']:>8.2f}s {r['input_tokens']:>8} "
          f"{r['output_tokens']:>8} ${r['cost']:>10.6f}")

sol = results[-1]
print(f"\n--- Relative to Sol ---")
for r in results:
    ratio = r["cost"] / sol["cost"] if sol["cost"] > 0 else 0
    print(f"  {r['model']:<18} {ratio:.1%} the cost")

print("""
--- GPT-5.6 Caching: what's different ---

Standard 4.1/5.4/5.5 caching:
  - Automatic prompt prefix caching (in-memory)
  - Cache read: ~10% of input rate
  - No charge for cache writes

GPT-5.6 caching (breakpoint model):
  - You mark cache breakpoints explicitly in your request
  - Minimum 30-minute cache lifetime (guaranteed retention)
  - Cache WRITES: 1.25x input rate (e.g. $5.00/M for Sol writes)
  - Cache READS: ~10% of input rate (e.g. $0.40/M for Sol reads)
  - Trade-off: pay more upfront for writes, save on repeated reads
  - Best for: long system prompts or tool definitions reused across many calls

Set breakpoints via:
  client.responses.create(
      model="gpt-5.6-sol",
      input=[
          {"role": "system", "content": "...", "cache_control": {"type": "breakpoint"}},
          {"role": "user", "content": "..."},
      ]
  )
""")

print("--- When to reach for each tier ---")
print("  gpt-5.6-luna  ($0.20/$1.20/M): High-volume classification, routing,")
print("                  simple extraction. 80% cheaper than original preview.")
print("  gpt-5.6-terra ($2.00/$12.00/M): Balanced workloads; replaces gpt-5.4-mini")
print("                  for most agentic flows at a competitive price point.")
print("  gpt-5.6-sol   ($4.00/$20.00/M): Hardest reasoning, coding, research.")
print("                  New default for high-quality production flows.")
print("                  Price cut Aug 2026: was $5.00/$30.00 at limited preview.")

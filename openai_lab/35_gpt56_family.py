"""Exercise 35: GPT-5.6 family — Sol, Terra, Luna (GA July 9, 2026).

GPT-5.6 is a three-tier model family that replaces GPT-5.5 as the flagship:

  gpt-5.6-sol   — $5.00/$30.00/M — Flagship. Hard reasoning, coding, cyber, agentic tasks.
  gpt-5.6-terra — $2.50/$15.00/M — Balanced. Premium production at roughly gpt-5.5 quality.
  gpt-5.6-luna  — $1.00/$6.00/M  — Lightweight. High-volume everyday tasks.

New cache billing in GPT-5.6 (differs from all prior families):
  - Cache READS:  10% of uncached input (same as before).
  - Cache WRITES: 1.25x the uncached input rate (new — was free for gpt-4.1/5.4/5.5).
  - Minimum cache lifetime: 30 minutes with explicit cache breakpoints.
  - Net: caching still wins if the same prefix is reused ≥ 2 times.

Other headline features (see platform.openai.com release notes for exact params):
  - Programmatic Tool Calling via the Responses API.
  - Multi-agent orchestration primitives (beta).
  - Persisted reasoning and max reasoning effort.
  - All three tiers: 1M context window, 128K max output, Feb 2026 knowledge cutoff.
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A startup has 500 GB of customer support transcripts in S3. "
    "They want a production-ready system where support managers can query "
    "the corpus in natural language ('What are the top 5 complaint themes this week?'). "
    "Design the end-to-end architecture: ingestion, embedding, retrieval, and the LLM "
    "query layer. Call out tradeoffs at each step."
)

# GPT-5.6 family (GA July 9, 2026)
MODELS = [
    "gpt-5.6-luna",   # Lightweight — cost-sensitive, high-volume
    "gpt-5.6-terra",  # Balanced — premium production
    "gpt-5.6-sol",    # Flagship — hardest tasks, best quality
]

# Pricing per 1M tokens (verified July 19, 2026)
# cached_read = cache read price (10% of input)
# cached_write = cache write price (1.25x input) — new for 5.6+
PRICING = {
    "gpt-5.6-luna":  {"input": 1.00, "output": 6.00,  "cached_read": 0.10,  "cached_write": 1.25},
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00, "cached_read": 0.25,  "cached_write": 3.125},
    "gpt-5.6-sol":   {"input": 5.00, "output": 30.00, "cached_read": 0.50,  "cached_write": 6.25},
}

results = []

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    start = time.time()
    response = client.responses.create(model=model, input=PROMPT)
    elapsed = time.time() - start

    p = PRICING[model]
    cost_input = (response.usage.input_tokens / 1_000_000) * p["input"]
    cost_output = (response.usage.output_tokens / 1_000_000) * p["output"]
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

# Tier comparison table
print("\n" + "=" * 60)
print("GPT-5.6 TIER COMPARISON")
print("=" * 60)
print(f"{'Model':<20} {'Latency':>8} {'In tok':>8} {'Out tok':>8} {'Cost':>12}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<20} {r['elapsed']:>7.2f}s {r['input_tokens']:>8} {r['output_tokens']:>8} ${r['cost']:>10.6f}")

sol = next(r for r in results if r["model"] == "gpt-5.6-sol")
print(f"\n--- Relative to gpt-5.6-sol (flagship) ---")
for r in results:
    cost_ratio = r["cost"] / sol["cost"] if sol["cost"] > 0 else 0
    print(f"  {r['model']:<20}  {cost_ratio:.0%} of Sol's cost")

# Cache write billing explainer
print("\n" + "=" * 60)
print("CACHE BILLING — new in GPT-5.6")
print("=" * 60)
print("""
For GPT-5.6+, cache writes are billed at 1.25x the model's uncached input rate.
Cache reads remain at 10% (same as gpt-4.1/5.4/5.5).

Worked example (gpt-5.6-sol, 50K-token system prompt, 1000 calls/day):

  Without caching:
    1000 calls × 50K tok × $5.00/M = $250.00/day

  With caching (1 write, 999 reads):
    Write:  50K × $6.25/M  = $0.31   (first call pays to warm the cache)
    Reads:  999 × 50K × $0.50/M = $24.98
    Total:                           $25.29/day  → 90% savings

  Breakeven on cache write vs uncached:
    write_cost / (uncached_per_call - read_per_call)
    = $0.31 / ($0.25 - $0.025)  ≈  1.4 calls

  Rule: if the same prefix is reused ≥ 2 times, caching still wins decisively.
        For one-shot requests, skip caching — the write costs more than uncached.

Minimum cache lifetime: 30 minutes (explicit breakpoints; prior models: 5–10 min).
""")

print("--- When to pick each tier ---")
print("gpt-5.6-luna  — Classification, routing, extraction, chat at volume. $1/$6/M.")
print("gpt-5.6-terra — Production workloads needing frontier quality. $2.50/$15/M.")
print("               Roughly gpt-5.5 quality at half the per-token price.")
print("gpt-5.6-sol   — Hardest: multi-step reasoning, coding, cybersecurity. $5/$30/M.")
print()
print("Migration from gpt-5.5:")
print("  gpt-5.6-terra is the natural upgrade — similar quality, 50% lower input cost.")
print("  gpt-5.6-sol is for tasks where 5.5 was noticeably struggling.")

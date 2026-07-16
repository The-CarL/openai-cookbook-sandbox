"""Exercise 35: GPT-5.6 family — Sol, Terra, Luna (GA July 9, 2026).

GPT-5.6 introduces three tiers that map to a new naming convention:
  Sol   — frontier capability (replaces the unsuffixed flagship role)
  Terra — balanced intelligence/cost (replaces the -mini tier)
  Luna  — cost-efficient, high-volume (replaces the -nano tier)

New in GPT-5.6 vs GPT-5.5:
  - Better reasoning, coding, and cybersecurity benchmarks
  - 1.05M context window (up from 1M) with 128K max output
  - Explicit cache breakpoints: control WHERE the 30-min cache boundary sits
  - Cache WRITES now billed at 1.25× uncached input (new charge vs prior models)
  - Cache reads: same 90% discount as always
  - Programmatic Tool Calling (define when/how tools may fire)
  - Persisted reasoning tokens (reuse reasoning across turns)
  - Max reasoning effort parameter
  - >272K input: billed at 2× input + 1.5× output (same long-context gotcha as 5.5)

Model aliases:
  "gpt-5.6"        — routes to gpt-5.6-sol (the flagship tier)
  "gpt-5.6-sol"    — explicit Sol
  "gpt-5.6-terra"  — explicit Terra
  "gpt-5.6-luna"   — explicit Luna
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Pricing per 1M tokens (verified July 9, 2026)
# Cache write: 1.25× uncached input rate (NEW in 5.6 — prior models don't charge writes)
# Cache read: 10% of uncached input rate (same 90% discount)
# Long-context (>272K input): 2× input + 1.5× output for the FULL session
PRICING = {
    "gpt-5.6-sol":   {"input": 5.00,  "output": 30.00, "cache_write": 6.25,  "cache_read": 0.50},
    "gpt-5.6-terra": {"input": 2.50,  "output": 15.00, "cache_write": 3.125, "cache_read": 0.25},
    "gpt-5.6-luna":  {"input": 1.00,  "output": 6.00,  "cache_write": 1.25,  "cache_read": 0.10},
}

PROMPT = (
    "A customer says: 'We're evaluating whether to migrate our agentic pipeline "
    "from GPT-5.5 to GPT-5.6 Sol. We run ~50K calls/day averaging 2K input + 800 "
    "output tokens. We use prompt caching heavily — about 60% cache hit rate. "
    "What should we know about GPT-5.6 caching economics before we switch?' "
    "Give a concise, structured answer."
)


# --- Example 1: Compare all three GPT-5.6 tiers ---

print("=" * 60)
print("EXAMPLE 1: GPT-5.6 Sol / Terra / Luna comparison")
print("=" * 60)

results = []

for model_id in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]:
    print(f"\n--- {model_id} ---")
    start = time.time()
    response = client.responses.create(model=model_id, input=PROMPT)
    elapsed = time.time() - start

    p = PRICING[model_id]
    cost_input = (response.usage.input_tokens / 1_000_000) * p["input"]
    cost_output = (response.usage.output_tokens / 1_000_000) * p["output"]
    cost_total = cost_input + cost_output

    results.append({
        "model": model_id,
        "elapsed": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": cost_total,
        "text": response.output_text,
    })

    print(response.output_text[:400], "..." if len(response.output_text) > 400 else "")
    print(f"  Latency: {elapsed:.2f}s | Tokens: {response.usage.input_tokens}in / {response.usage.output_tokens}out | Cost: ${cost_total:.6f}")

print("\n" + "=" * 60)
print("TIER COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<18} {'Latency':>8} {'In tok':>8} {'Out tok':>8} {'Cost':>12}")
print("-" * 60)
for r in results:
    print(f"{r['model']:<18} {r['elapsed']:>7.2f}s {r['input_tokens']:>8} {r['output_tokens']:>8} ${r['cost']:>10.6f}")


# --- Example 2: Cache-write economics (new in GPT-5.6) ---

print("\n" + "=" * 60)
print("EXAMPLE 2: Cache-write economics (new in GPT-5.6)")
print("=" * 60)
print("""
GPT-5.5 and earlier: cache WRITES were free. You paid only for reads.
GPT-5.6: cache WRITES cost 1.25× the uncached input rate.

This changes the break-even math for caching:
  - If you write a 10K-token prompt prefix once and read it N times,
    you need enough reads to offset the write surcharge.

Break-even formula:
  write_cost  = prefix_tokens × cache_write_rate
  read_saving = prefix_tokens × (input_rate - cache_read_rate) per hit
  break_even  = write_cost / read_saving  (number of cache hits needed)
""")

model = "gpt-5.6-sol"
p = PRICING[model]
prefix_tokens = 5_000
write_cost = (prefix_tokens / 1_000_000) * p["cache_write"]
saving_per_hit = (prefix_tokens / 1_000_000) * (p["input"] - p["cache_read"])
break_even_hits = write_cost / saving_per_hit if saving_per_hit > 0 else float("inf")

print(f"Model: {model}")
print(f"Prefix size: {prefix_tokens:,} tokens")
print(f"Cache write cost:       ${write_cost:.6f}  ({prefix_tokens:,} tokens × ${p['cache_write']:.4f}/M)")
print(f"Saving per cache read:  ${saving_per_hit:.6f}  ({prefix_tokens:,} tokens × (${p['input']}-${p['cache_read']})/M)")
print(f"Break-even at:          {break_even_hits:.2f} cache hits")
print(f"\nConclusion: if you reuse this prefix more than {break_even_hits:.0f} times, caching saves money.")
print("At typical >100× reuse in production, caching is still overwhelmingly worth it.")


# --- Example 3: Migration check — GPT-5.5 vs GPT-5.6-Sol ---

print("\n" + "=" * 60)
print("EXAMPLE 3: GPT-5.5 vs GPT-5.6-Sol on the same prompt")
print("=" * 60)

compare_prompt = (
    "What are three key considerations when choosing between in-memory "
    "and extended prompt caching for a production RAG system? Be concise."
)

for model_id in ["gpt-5.5", "gpt-5.6-sol"]:
    start = time.time()
    r = client.responses.create(model=model_id, input=compare_prompt)
    elapsed = time.time() - start
    print(f"\n{model_id} ({elapsed:.2f}s, {r.usage.input_tokens}in/{r.usage.output_tokens}out):")
    print(r.output_text)


# --- Summary ---

print("\n" + "=" * 60)
print("GPT-5.6 KEY FACTS (July 9, 2026)")
print("=" * 60)
print("""
Tier       Model ID         Input $/M   Output $/M  Cache write $/M  Cache read $/M
─────────────────────────────────────────────────────────────────────────────────────
Sol        gpt-5.6-sol      5.00        30.00        6.25             0.50
Terra      gpt-5.6-terra    2.50        15.00        3.125            0.25
Luna       gpt-5.6-luna     1.00         6.00        1.25             0.10

Alias: "gpt-5.6" → gpt-5.6-sol

Context: 1.05M tokens in, 128K tokens out (all three tiers)

New caching model vs GPT-5.5:
  - Cache writes are now billed at 1.25× input rate (was free in ≤5.5)
  - Cache reads retain the 90% discount (unchanged)
  - Explicit cache breakpoints: you control WHERE in the prompt the boundary sits
  - Minimum cache lifetime: 30 minutes (unchanged from 5.5)

Long-context pricing (same rule as GPT-5.5):
  Sessions with >272K input tokens billed at 2× input + 1.5× output for the FULL session.

When to reach for each tier:
  Sol   — hardest reasoning/coding tasks, cybersecurity, when quality is the constraint
  Terra — default for new agentic workloads; matches 5.4 per-token price with better quality
  Luna  — high-volume production at minimum cost; step up to Terra when quality matters
""")

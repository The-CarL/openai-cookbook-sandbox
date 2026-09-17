"""Exercise 35: GPT-5.6 family — Sol/Terra/Luna, explicit caching, xhigh/max reasoning.

GPT-5.6 went GA on July 9, 2026. Three tiers with 1.05M-token context window.
Key new capabilities vs earlier families:
  - Explicit prompt caching: prompt_cache_options.mode="explicit" + ttl (not in-memory)
    Cache writes billed at 1.25x input; reads at 10% of input; 30-min minimum TTL.
  - Six-level reasoning effort: none / low / medium / high / xhigh / max
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Pricing reference (September 17, 2026) ---
# GPT-5.6 Sol promotional pricing through November 21, 2026.
# Terra/Luna prices were cut July 30, 2026.
PRICING = {
    "gpt-5.6-luna":  {"input": 0.20, "output": 1.20,  "cached_input": 0.02},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00, "cached_input": 0.20},
    "gpt-5.6-sol":   {"input": 4.00, "output": 20.00, "cached_input": 0.40},
}

# 1.25x write surcharge applies to cache writes; see Example 2.
CACHE_WRITE_MULTIPLIER = 1.25


def cost(response) -> float:
    model = response.model.split("-202")[0]  # strip dated snapshot suffix
    p = PRICING.get(model, {})
    if not p:
        return 0.0
    u = response.usage
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)
    non_cached = u.input_tokens - cached
    return (
        (non_cached / 1_000_000) * p["input"]
        + (cached / 1_000_000) * p["cached_input"]
        + (u.output_tokens / 1_000_000) * p["output"]
    )


HARD_PROMPT = (
    "A distributed system has three microservices — Auth, Order, and Inventory — "
    "communicating via gRPC. Under 10k RPS the system sees ~0.1% 503s on the Order "
    "service, but only when Auth latency spikes above 200ms. Inventory is unaffected. "
    "Design a step-by-step root-cause investigation plan and propose two remediation "
    "strategies (one short-term, one architectural) with trade-off analysis."
)


# === Example 1: Sol / Terra / Luna comparison on a hard prompt ===
print("=" * 60)
print("EXAMPLE 1: Sol / Terra / Luna — quality vs cost on a hard prompt")
print("=" * 60)
print()

results = []
for model in ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]:
    import time
    t0 = time.time()
    response = client.responses.create(model=model, input=HARD_PROMPT)
    elapsed = time.time() - t0
    c = cost(response)
    results.append({"model": model, "elapsed": elapsed, "cost": c, "text": response.output_text})
    print(f"\n{'='*40}")
    print(f"MODEL: {model}")
    print(f"Latency: {elapsed:.2f}s | Cost: ${c:.6f}")
    print(f"\n{response.output_text[:600]}...")

print("\n--- Summary ---")
print(f"{'Model':<18} {'Latency':>9} {'Cost':>12}")
for r in results:
    print(f"{r['model']:<18} {r['elapsed']:>8.2f}s ${r['cost']:>10.6f}")


# === Example 2: Explicit prompt caching ===
print()
print("=" * 60)
print("EXAMPLE 2: Explicit prompt caching (GPT-5.6 only)")
print("=" * 60)
print()
print("GPT-5.6 uses EXPLICIT prompt caching, not in-memory caching.")
print("You opt in per-request via prompt_cache_options.")
print()
print("  Cache writes: 1.25x input rate (you pay a small surcharge to store)")
print("  Cache reads:  10% of input rate (90% off — same savings as in-memory)")
print("  Minimum TTL:  30 minutes (set shorter TTLs if you want it evicted sooner)")
print()

# A long system prompt that's worth caching across many calls.
SYSTEM_PROMPT = (
    "You are an expert software architect specializing in distributed systems. "
    "You have deep knowledge of: microservices patterns, gRPC and REST APIs, "
    "Kubernetes, service mesh (Istio/Linkerd), distributed tracing (Jaeger/Zipkin), "
    "APM tools (Datadog, New Relic), circuit breaker patterns (Hystrix, resilience4j), "
    "load balancing strategies, and database connection pooling. "
    "Always structure your answers with: (1) root cause hypothesis, "
    "(2) investigation steps, (3) short-term fix, (4) long-term architectural change. "
    * 5  # repeat to make the prompt long enough to cache
)

# First call — cache write (pays 1.25x surcharge)
print("--- Call 1: Cache write (warm the cache) ---")
r_write = client.responses.create(
    model="gpt-5.6-sol",
    instructions=SYSTEM_PROMPT,
    input=HARD_PROMPT,
    prompt_cache_options={"mode": "explicit", "ttl": 3600},  # cache for 1 hour
)
cached_write = getattr(getattr(r_write.usage, "input_tokens_details", None), "cached_tokens", 0)
print(f"Input tokens:   {r_write.usage.input_tokens}")
print(f"Cached tokens:  {cached_write}  (0 on first call — cache was cold)")
print(f"Output tokens:  {r_write.usage.output_tokens}")
print(f"Est. cost:      ${cost(r_write):.6f}")
print()
print("NOTE: On the write call, the cache stores your system prompt.")
print("      The write surcharge (1.25x) applies to the tokens being stored.")
print()

# Second call — cache read (pays 10% = 90% off)
print("--- Call 2: Cache read (same system prompt, different question) ---")
r_read = client.responses.create(
    model="gpt-5.6-sol",
    instructions=SYSTEM_PROMPT,
    input="What is the most common cause of gRPC deadline exceeded errors in production?",
    prompt_cache_options={"mode": "explicit", "ttl": 3600},
)
cached_read = getattr(getattr(r_read.usage, "input_tokens_details", None), "cached_tokens", 0)
print(f"Input tokens:   {r_read.usage.input_tokens}")
print(f"Cached tokens:  {cached_read}  (should be > 0 if cache hit)")
print(f"Output tokens:  {r_read.usage.output_tokens}")
print(f"Est. cost:      ${cost(r_read):.6f}")
if cached_read > 0:
    savings = (cached_read / 1_000_000) * (PRICING["gpt-5.6-sol"]["input"] - PRICING["gpt-5.6-sol"]["cached_input"])
    print(f"Savings vs uncached: ${savings:.6f}")


# === Example 3: xhigh and max reasoning effort ===
print()
print("=" * 60)
print("EXAMPLE 3: xhigh and max reasoning effort (new in GPT-5.6)")
print("=" * 60)
print()
print("GPT-5.6 adds xhigh and max effort levels beyond the previous high cap.")
print("Use xhigh for hard math/code/logic. Use max for the hardest problems.")
print()

MATH_PROBLEM = (
    "Prove that for any prime p > 3, p² − 1 is always divisible by 24. "
    "Then generalize: for which values of n is n² − 1 divisible by 24?"
)

for effort in ["medium", "high", "xhigh"]:
    r = client.responses.create(
        model="gpt-5.6-sol",
        input=MATH_PROBLEM,
        reasoning={"effort": effort},
    )
    reasoning_tokens = getattr(getattr(r.usage, "output_tokens_details", None), "reasoning_tokens", 0)
    print(f"Effort={effort:<7}: {r.usage.output_tokens:>5} output tokens "
          f"({reasoning_tokens} reasoning), cost=${cost(r):.5f}")
    print(f"  Preview: {r.output_text[:200].strip()}...")
    print()


# === Summary ===
print("=" * 60)
print("GPT-5.6 QUICK REFERENCE")
print("=" * 60)
print("""
Model IDs:  gpt-5.6-luna | gpt-5.6-terra | gpt-5.6-sol
            (bare "gpt-5.6" aliases to Sol)

Context:    1.05M tokens across all three tiers

Pricing (Sep 17, 2026):
  luna:   $0.20 in / $1.20 out  (cache read: $0.02/M)
  terra:  $2.00 in / $12.00 out (cache read: $0.20/M)
  sol:    $4.00 in / $20.00 out (cache read: $0.40/M) — promo thru Nov 21

Explicit prompt caching (GPT-5.6 only):
  opt-in: prompt_cache_options={"mode": "explicit", "ttl": <seconds>}
  write:  1.25x input rate (surcharge to store)
  read:   10% of input rate (90% savings)
  min TTL: 1800s (30 minutes)
  contrast: GPT-4.1/5.4/5.5 use automatic in-memory caching at 10% rate, no write cost

Reasoning effort levels (6 vs 3 in earlier families):
  none / low / medium / high / xhigh / max
  Use xhigh/max for difficult proofs, complex codegen, adversarial reasoning.

When to use which tier:
  luna:  Budget tier — classification, routing, cheap extraction. Replaces 4.1-nano.
  terra: Everyday quality. Competes with GPT-5.5 at lower price. Default for most flows.
  sol:   Hardest tasks, frontier reasoning. Promo pricing makes it competitive with 5.5.
""")

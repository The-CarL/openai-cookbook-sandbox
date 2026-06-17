"""Exercise 25: Prompt caching — production cost optimization.

Cached input tokens cost ~10% of regular input tokens. For workloads with a
shared system prompt, large tool list, or RAG context, caching is the single
biggest cost lever. This exercise shows the cache lifecycle and how to verify
hits via usage.input_tokens_details.cached_tokens.
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# A long, stable system prompt is the prefix that benefits from caching.
# Caching keys on the prefix, so anything that varies (the user question)
# must come AFTER the stable content. Minimum cacheable prefix is 1024 tokens.
STABLE_SYSTEM_PROMPT = """You are a senior solutions engineer for CloudSync,
an enterprise data sync platform. Answer questions using the reference
material below. Be concise, cite the relevant section, and flag anything
that needs Sales involvement.

=== PRODUCT REFERENCE (cached) ===
""" + ("""
CloudSync Enterprise Platform — Section %d of 20
- Real-time bidirectional sync across 40+ enterprise systems
- AES-256 at rest, TLS 1.3 in transit, customer-managed keys on Enterprise
- Compliance: SOC 2 Type II, HIPAA BAA, GDPR DPA, ISO 27001, FedRAMP Moderate
- Data residency: US, EU, UK, Japan, Australia (Enterprise tier only)
- Pricing tiers: Starter $5K/mo (1M records/day), Professional $15K/mo
  (10M records/day), Enterprise custom (unlimited, dedicated support)
- Connectors: Salesforce, SAP, Oracle, Workday, ServiceNow, Snowflake,
  BigQuery, Databricks, Epic, Workday HR, NetSuite
- SLA: 99.99% uptime, p99 latency under 500ms
- Support: 24/7 SOC, MTTD <15min, MTTR <1hr for P1, BAA-required notify <72hr
""" * 1 for _ in range(15))  # ~15x repetition to safely exceed the 1024-token minimum


def cached_call(question: str):
    """Make a Responses call where the cacheable prefix comes first."""
    return client.responses.create(
        model="gpt-4.1-mini",
        instructions=STABLE_SYSTEM_PROMPT,  # The cacheable prefix
        input=question,                     # The varying suffix
    )


def report(label: str, response, elapsed: float):
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    fresh = usage.input_tokens - cached
    hit_rate = cached / usage.input_tokens if usage.input_tokens else 0

    # Pricing math (gpt-4.1-mini): $0.40/M input, $0.10/M cached input, $1.60/M output
    cost_fresh = fresh / 1_000_000 * 0.40
    cost_cached = cached / 1_000_000 * 0.10
    cost_output = usage.output_tokens / 1_000_000 * 1.60
    total = cost_fresh + cost_cached + cost_output
    # What it would have cost with no caching
    cost_uncached = usage.input_tokens / 1_000_000 * 0.40 + cost_output

    print(f"\n--- {label} ---")
    print(f"  Latency:        {elapsed*1000:>6.0f}ms")
    print(f"  Input tokens:   {usage.input_tokens:>6} ({fresh} fresh, {cached} cached, {hit_rate:.0%} hit)")
    print(f"  Output tokens:  {usage.output_tokens:>6}")
    print(f"  Cost:           ${total:.6f}  (vs ${cost_uncached:.6f} uncached, saved ${cost_uncached - total:.6f})")


print("=" * 70)
print("PROMPT CACHING — measuring hits across repeated calls")
print("=" * 70)
print(f"\nSystem prompt is ~{len(STABLE_SYSTEM_PROMPT) // 4} tokens (rough estimate).")
print("Each call uses the same prefix but a different question.")

questions = [
    "Does CloudSync support Epic EHR integration?",
    "What's the SLA for p99 latency, and what happens if it's breached?",
    "Can a healthcare customer get a HIPAA BAA on the Professional tier?",
    "Which compliance certifications does CloudSync hold today?",
    "What's the price difference between Starter and Professional?",
]

# First call: cache MISS (cold). Subsequent calls: cache HIT.
for i, q in enumerate(questions, 1):
    start = time.time()
    response = cached_call(q)
    elapsed = time.time() - start
    report(f"Call {i}: {q[:55]}...", response, elapsed)

# --- Demonstrating cache invalidation ---
print("\n" + "=" * 70)
print("CACHE INVALIDATION — change the prefix, lose the cache")
print("=" * 70)
print("\nAdding a single character to the system prompt invalidates the cache.")

modified_prompt = STABLE_SYSTEM_PROMPT + "\n=== UPDATED 2026-04-27 ==="

start = time.time()
response = client.responses.create(
    model="gpt-4.1-mini",
    instructions=modified_prompt,
    input="Does CloudSync support Epic EHR integration?",
)
elapsed = time.time() - start
report("After modifying the prefix", response, elapsed)

print("\n" + "=" * 70)
print("PRODUCTION CACHING CHECKLIST")
print("=" * 70)
print("""
1. Put STABLE content first (system prompt, tool defs, RAG corpus, few-shot
   examples). Put VARIABLE content (user message) last.

2. Cache key is computed on the EXACT prefix bytes. Any change — even a
   timestamp injection or shuffled tool order — invalidates it.

3. Minimum cacheable prefix is 1024 tokens. Below that, no caching happens.

4. Cache retention (June 2026 change): orgs *without* ZDR enabled now get
   24h extended caching by default (prompt_cache_retention='24h'). ZDR orgs
   still default to in_memory (~5–10 min idle). Pass
   prompt_cache_retention='24h' explicitly on ZDR plans to opt in to
   extended retention. Check your org settings at platform.openai.com.

5. Verify hits via response.usage.input_tokens_details.cached_tokens.
   Do NOT assume — instrument it. Cache hit rate is a top-line cost metric.

6. Pricing rule of thumb: cached input = 10% of regular input. A 90% hit
   rate on a 10K-token system prompt cuts input cost by ~80%.

7. Don't try to outsmart caching with manual chunking. Just make the prefix
   stable. The infra handles the rest.

8. Pairs naturally with context compaction (Exercise 26): keep the long-term
   context cached, compact the conversation history.
""")

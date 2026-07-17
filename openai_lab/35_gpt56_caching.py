"""Exercise 35: GPT-5.6 caching — prompt_cache_key, prompt_cache_options, cache write billing.

GPT-5.6 (GA July 9, 2026) introduces a new caching model distinct from all
earlier families:

  prompt_cache_key      Required for reliable cache matching across sessions.
                        Without it, cache matching is best-effort.

  prompt_cache_options  Request-wide cache policy:
    .mode = "implicit"  (default) — OpenAI picks cache boundaries automatically.
    .mode = "explicit"  — only the breakpoints you place are used.
    .ttl  = "30m" | "1h" | "24h"  — minimum cache lifetime (default: "30m").

  Cache write billing:  1.25× uncached input rate  (new — writes were FREE on 5.5 and earlier)
  Cache read discount:  0.10× uncached input rate  (same 90% discount as before)

  usage.input_tokens_details.cache_write_tokens  tokens written to cache this call
  usage.input_tokens_details.cached_tokens       tokens read from cache this call

  Break-even: pay 1.25× once to write; save 0.90× on every subsequent read hit.
  A prefix that gets hit ≥ 2 times pays for itself.

  prompt_cache_retention (used by 5.5 and earlier) is REPLACED by
  prompt_cache_options.ttl for GPT-5.6 and later.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

SYSTEM_PROMPT = (
    "You are an expert solutions engineer specializing in the OpenAI Responses API. "
    "Give concise, accurate answers with relevant API parameter names. "
    "Always include pricing and billing implications when discussing caching."
)

# --- Example 1: Implicit caching with prompt_cache_key ---
print("=" * 60)
print("EXAMPLE 1: Implicit caching with prompt_cache_key")
print("=" * 60)
print()
print("First call — cache COLD: tokens are written (1.25x billing)")
print()

r1 = client.responses.create(
    model="gpt-5.6-sol",
    prompt_cache_key="ex35-system-v1",
    prompt_cache_options={"mode": "implicit"},
    instructions=SYSTEM_PROMPT,
    input="What is prompt caching and how does it work in the Responses API?",
)

u1 = r1.usage
details1 = u1.input_tokens_details
wrote1 = getattr(details1, "cache_write_tokens", 0) if details1 else 0
read1  = getattr(details1, "cached_tokens", 0) if details1 else 0

print(f"Response: {r1.output_text[:200]}")
print(f"\nToken breakdown:")
print(f"  Input total:     {u1.input_tokens}")
print(f"  Cache writes:    {wrote1}  ← billed at 1.25× input rate")
print(f"  Cached reads:    {read1}  ← should be 0 on first call (cold cache)")
print(f"  Output tokens:   {u1.output_tokens}")
print()

print("Second call — same key, different question: cache WARM (read hits)")
print()

r2 = client.responses.create(
    model="gpt-5.6-sol",
    prompt_cache_key="ex35-system-v1",
    prompt_cache_options={"mode": "implicit"},
    instructions=SYSTEM_PROMPT,
    input="Explain the 30-minute minimum cache lifetime.",
)

u2 = r2.usage
details2 = u2.input_tokens_details
wrote2 = getattr(details2, "cache_write_tokens", 0) if details2 else 0
read2  = getattr(details2, "cached_tokens", 0) if details2 else 0

print(f"Response: {r2.output_text[:200]}")
print(f"\nToken breakdown:")
print(f"  Input total:     {u2.input_tokens}")
print(f"  Cache writes:    {wrote2}  ← should be 0 or minimal on cache hit")
print(f"  Cached reads:    {read2}  ← discounted at 0.10× input rate")
print(f"  Output tokens:   {u2.output_tokens}")
print()

# --- Example 2: Explicit cache breakpoints ---
print("=" * 60)
print("EXAMPLE 2: Explicit cache breakpoints (mode='explicit')")
print("=" * 60)
print()
print("Place breakpoints after stable prefixes you want to lock in as cache boundaries.")
print("In explicit mode the model ONLY uses the breakpoints you mark.")
print()

STABLE_CONTEXT = (
    "You are an OpenAI API expert.\n\n"
    "Customer context (stable, reused across all API calls):\n"
    "  - Customer: Acme Corp, enterprise tier, $480K/yr\n"
    "  - Use case: Internal support chatbot with 500K calls/day\n"
    "  - Key constraint: minimize per-call cost at scale\n"
    "  - Models in use: gpt-5.6-luna (routing), gpt-5.6-sol (complex queries)\n"
)

r3 = client.responses.create(
    model="gpt-5.6-sol",
    prompt_cache_key="ex35-explicit-v1",
    prompt_cache_options={"mode": "explicit", "ttl": "1h"},
    input=[
        {
            "type": "input_text",
            "text": STABLE_CONTEXT,
            "cache_control": {"type": "breakpoint"},
        },
        {
            "type": "input_text",
            "text": "Should Acme route all queries to gpt-5.6-luna or split by complexity?",
        },
    ],
)

u3 = r3.usage
details3 = u3.input_tokens_details
wrote3 = getattr(details3, "cache_write_tokens", 0) if details3 else 0
read3  = getattr(details3, "cached_tokens", 0) if details3 else 0

print(f"Response: {r3.output_text[:250]}")
print(f"\nToken breakdown (first explicit call — writes the breakpoint prefix):")
print(f"  Input total:     {u3.input_tokens}")
print(f"  Cache writes:    {wrote3}  ← context before breakpoint billed at 1.25×")
print(f"  Cached reads:    {read3}  ← should be 0 on first call")
print(f"  Output tokens:   {u3.output_tokens}")
print()

# --- Example 3: Cost math ---
print("=" * 60)
print("EXAMPLE 3: Break-even calculator")
print("=" * 60)

INPUT_RATE = 5.00  # gpt-5.6-sol $/1M
WRITE_RATE = INPUT_RATE * 1.25
READ_RATE  = INPUT_RATE * 0.10

print(f"""
GPT-5.6 Sol cache economics (per 1M tokens):
  Standard input:  ${INPUT_RATE:.2f}
  Cache write:     ${WRITE_RATE:.2f}  (1.25 × ${INPUT_RATE:.2f} — more expensive than uncached!)
  Cache read:      ${READ_RATE:.2f}  (0.10 × ${INPUT_RATE:.2f} — 90% discount)

Break-even: pay extra {WRITE_RATE - INPUT_RATE:.2f} once to write; save {INPUT_RATE - READ_RATE:.2f} on each read.
  Writes-to-break-even = {(WRITE_RATE - INPUT_RATE) / (INPUT_RATE - READ_RATE):.1f}
  → A cached prefix pays for itself after just 2 requests that hit it.
""")

# Print for all three tiers
print(f"{'Model':<15} {'Input':>8} {'Write':>8} {'Read':>8} {'Break-even calls':>18}")
print("-" * 55)
for name, inp, out in [("gpt-5.6-sol", 5.00, 30.00), ("gpt-5.6-terra", 2.50, 15.00), ("gpt-5.6-luna", 1.00, 6.00)]:
    w = inp * 1.25
    r = inp * 0.10
    be = (w - inp) / (inp - r)
    print(f"{name:<15} ${inp:>6.2f}  ${w:>6.3f}  ${r:>6.3f}  {be:>15.1f}")

# --- Summary ---
print()
print("=" * 60)
print("GPT-5.6 CACHING KEY CONCEPTS")
print("=" * 60)
print("""
New vs. earlier models:
  Earlier (5.5 and before)        GPT-5.6 and later
  ──────────────────────────────  ──────────────────────────────────────
  Implicit caching only           Implicit (default) OR explicit
  Cache writes free               Cache writes billed at 1.25× input
  No prompt_cache_key             prompt_cache_key required for reliability
  prompt_cache_retention param    prompt_cache_options.ttl ("30m"/"1h"/"24h")
  No cache_write_tokens           cache_write_tokens in usage details

API call with explicit breakpoints:
  response = client.responses.create(
      model="gpt-5.6-sol",
      prompt_cache_key="stable-prefix-v1",      # key for reliable matching
      prompt_cache_options={
          "mode": "explicit",                    # only your breakpoints count
          "ttl": "1h",                           # 30m minimum regardless
      },
      input=[
          {"type": "input_text", "text": STABLE_PREFIX,
           "cache_control": {"type": "breakpoint"}},   # write boundary here
          {"type": "input_text", "text": user_question},
      ],
  )

  # Read write vs. read counts from usage
  details = response.usage.input_tokens_details
  wrote = details.cache_write_tokens   # billed at 1.25× input rate
  read  = details.cached_tokens        # billed at 0.10× input rate

Cache decision guide:
  High reuse (same prefix > 2 calls)  → cache writes pay for themselves quickly
  Low reuse (one-off requests)        → avoid cache; write cost hurts you
  Best candidates: system prompts, customer context, long RAG boilerplate
""")

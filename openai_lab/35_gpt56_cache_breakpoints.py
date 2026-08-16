"""Exercise 35: GPT-5.6 explicit prompt-cache breakpoints.

GPT-5.6 (Sol/Terra/Luna, GA July 9, 2026) changes the caching model in three
important ways compared to GPT-5.5 and earlier:

1. WRITES NOW COST 1.25× the uncached input rate (reads still cost 10%).
   With GPT-4.1/5.4/5.5 every cache write was free; now it is not.

2. EXPLICIT BREAKPOINTS — you can pin exactly which prefixes to cache using
   prompt_cache_options.mode="explicit" and the prompt_cache_breakpoint flag.
   Implicit caching still works but you no longer get free writes, so explicit
   control over what gets written pays off.

3. 30-MINUTE MINIMUM LIFETIME — cache entries persist for at least 30 minutes
   regardless of traffic. This makes bursty workloads cheaper because the old
   5-10 min idle eviction no longer kicks in between requests.

New usage field to instrument:
  response.usage.input_tokens_details.cache_write_tokens  (new in GPT-5.6)
  response.usage.input_tokens_details.cached_tokens       (same as before)

Pricing for gpt-5.6-sol (verified August 16, 2026, after July 30 cuts):
  Regular input:  $5.00 / 1M
  Cache WRITE:    $6.25 / 1M  (1.25× input)
  Cache READ:     $0.50 / 1M  (10% of input, same rule as before)
  Output:        $30.00 / 1M
"""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Pricing for gpt-5.6 family after July 30, 2026 cuts.
GPT56_PRICING = {
    "gpt-5.6-sol": {
        "input": 5.00, "output": 30.00,
        "cache_write": 6.25,   # 1.25× input
        "cache_read": 0.50,    # 10% of input
    },
    "gpt-5.6-terra": {
        "input": 2.00, "output": 12.00,
        "cache_write": 2.50,
        "cache_read": 0.20,
    },
    "gpt-5.6-luna": {
        "input": 0.20, "output": 1.20,
        "cache_write": 0.25,
        "cache_read": 0.02,
    },
}

# Large, stable prefix that benefits from caching.
SYSTEM_PROMPT = """You are a senior API solutions engineer for Acme Corp.
Answer concisely using the product reference below.

=== PRODUCT REFERENCE (cache this) ===
""" + (
    "Section content — product specs, pricing tiers, SLA details, "
    "compliance certifications, integration guide, FAQ entries.\n"
) * 80  # ~2 000 tokens; safely above the 1 024-token cache minimum


def cost_report(label: str, response, prices: dict) -> None:
    """Print a cost breakdown including cache-write billing."""
    u = response.usage
    details = u.input_tokens_details if u.input_tokens_details else None

    cached_read   = getattr(details, "cached_tokens", 0) if details else 0
    cache_writes  = getattr(details, "cache_write_tokens", 0) if details else 0
    fresh_input   = u.input_tokens - cached_read - cache_writes

    write_cost   = cache_writes / 1_000_000 * prices["cache_write"]
    read_cost    = cached_read  / 1_000_000 * prices["cache_read"]
    fresh_cost   = fresh_input  / 1_000_000 * prices["input"]
    output_cost  = u.output_tokens / 1_000_000 * prices["output"]
    total        = write_cost + read_cost + fresh_cost + output_cost

    # What the same call would cost with no caching at all
    no_cache = u.input_tokens / 1_000_000 * prices["input"] + output_cost

    print(f"\n--- {label} ---")
    print(f"  Input tokens:   {u.input_tokens:>6}  "
          f"({fresh_input} fresh, {cache_writes} write, {cached_read} read)")
    print(f"  Output tokens:  {u.output_tokens:>6}")
    print(f"  Cost:  fresh=${fresh_cost:.5f}  "
          f"write=${write_cost:.5f}  read={read_cost:.5f}  out=${output_cost:.5f}"
          f"  TOTAL=${total:.5f}")
    print(f"  vs no-cache: ${no_cache:.5f}  "
          f"(net {'saved' if no_cache > total else 'extra'}: "
          f"${abs(no_cache - total):.5f})")


MODEL = "gpt-5.6-sol"
prices = GPT56_PRICING[MODEL]
questions = [
    "What compliance certifications does Acme Corp hold?",
    "What's the SLA for API latency?",
    "Which enterprise tiers include customer-managed keys?",
]

print("=" * 70)
print("GPT-5.6 EXPLICIT CACHE BREAKPOINTS")
print("=" * 70)

# ── Pattern 1: IMPLICIT mode (default) ──────────────────────────────────────
# Works the same as GPT-5.5, but writes are now billed at 1.25× input.
print("\n" + "─" * 70)
print("Pattern 1 — Implicit caching (default, writes now billed)")
print("─" * 70)

for i, q in enumerate(questions, 1):
    t0 = time.time()
    r = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=q,
    )
    elapsed = time.time() - t0
    cost_report(f"Implicit call {i} ({elapsed*1000:.0f}ms)", r, prices)

# ── Pattern 2: EXPLICIT mode — pin exactly what to cache ────────────────────
# Use prompt_cache_options.mode="explicit" and mark stable content with
# prompt_cache_breakpoint=True so writes happen only on the prefix you choose.
# Avoids accidental writes to prefixes you don't intend to pin.
print("\n" + "─" * 70)
print("Pattern 2 — Explicit breakpoints (pin the stable prefix only)")
print("─" * 70)
print("prompt_cache_options.mode='explicit' + prompt_cache_breakpoint=True")

for i, q in enumerate(questions, 1):
    t0 = time.time()
    r = client.responses.create(
        model=MODEL,
        # Explicit mode: only content tagged prompt_cache_breakpoint=True
        # will be written to cache. Reads still work on previously written entries.
        prompt_cache_options={"mode": "explicit"},
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_PROMPT,
                        # This is the only point we want written to cache.
                        # On call 1 it pays 1.25× to write; calls 2+ read at 10%.
                        "prompt_cache_breakpoint": True,
                    }
                ],
            },
            {"role": "user", "content": q},
        ],
    )
    elapsed = time.time() - t0
    cost_report(f"Explicit call {i} ({elapsed*1000:.0f}ms)", r, prices)

# ── Pattern 3: Custom TTL ────────────────────────────────────────────────────
# prompt_cache_options.ttl controls how long to keep the cache entry.
# GPT-5.6 minimum is 1800s (30 min); you can set longer for hot prefixes.
# (prompt_cache_retention from older models is replaced by this field.)
print("\n" + "─" * 70)
print("Pattern 3 — Custom TTL (keep cache for 2 hours)")
print("─" * 70)

r = client.responses.create(
    model=MODEL,
    prompt_cache_options={"mode": "explicit", "ttl": 7200},  # 2 hours
    input=[
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": SYSTEM_PROMPT,
                    "prompt_cache_breakpoint": True,
                }
            ],
        },
        {"role": "user", "content": questions[0]},
    ],
)
cost_report("Custom-TTL call", r, prices)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GPT-5.6 CACHING CHEAT-SHEET")
print("=" * 70)
print(f"""
Model: {MODEL}  (after July 30, 2026 prices — Sol unchanged, Terra -20%, Luna -80%)

Pricing:
  Regular input:   ${prices['input']:.2f} / 1M tokens
  Cache WRITE:     ${prices['cache_write']:.2f} / 1M  (NEW: 1.25× input — was free on 5.5)
  Cache READ:      ${prices['cache_read']:.2f} / 1M  (still 10% of input)
  Output:         ${prices['output']:.2f} / 1M tokens

Key differences vs GPT-5.5:
  - Cache WRITES are billed. Budget them like any other token type.
  - Minimum cache lifetime is 30 min (vs 5-10 min idle eviction on 5.5).
  - Explicit breakpoints let you pay to write ONLY what you intend to keep.
  - New usage field: input_tokens_details.cache_write_tokens (instrument this).
  - prompt_cache_retention is deprecated → use prompt_cache_options.ttl.

When explicit mode pays off:
  - Long, stable system prompts you reuse across many calls (amortise write cost).
  - Multi-turn agents where the tool list / RAG chunk grows but the prefix stays fixed.
  - Bursty workloads: 30-min minimum TTL means you pay the write cost once and
    read cheaply even during inter-request gaps that would have evicted 5.5 cache.

When to SKIP explicit mode:
  - One-off calls — paying 1.25× to write a cache entry you read once nets negative.
  - Prefixes under ~5 000 tokens — write cost vs read savings break-even takes many calls.
  - Dynamic prefixes that change every call (implicit or explicit, cache is useless).

Break-even calculation (Sol):
  Write saves $5.00 - $0.50 = $4.50 per 1M tokens per read.
  Write costs an extra $1.25 per 1M (1.25× vs $1.00 baseline).
  Break-even: 1.25 / 4.50 ≈ 0.28 reads — you need just ONE cached read per write.
  Luna break-even is identical in ratio; absolute savings are ~25× smaller.
""")

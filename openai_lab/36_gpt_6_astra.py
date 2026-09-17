"""Exercise 36: GPT-6 Astra — OpenAI's September 2026 frontier model.

GPT-6 Astra (model ID: gpt-6-astra) was released September 3–4, 2026.
It is OpenAI's most capable model, designed for the hardest end-to-end work:
reasoning, coding, computer use, research, and long-horizon document creation.

Pricing (September 2026):
  Input:        $10.00 / 1M tokens
  Output:       $50.00 / 1M tokens
  Cached input: $1.00 / 1M tokens (10% of input — standard rate)
  Batch / Flex: 50% off ($5/$25 per 1M)
  Fast mode:    2x rate ($20/$100 per 1M)

Context window: 1M tokens.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PRICING = {
    "gpt-6-astra":   {"input": 10.00, "output": 50.00, "cached_input": 1.00},
    # Prior flagships for comparison
    "gpt-5.6-sol":   {"input": 4.00,  "output": 20.00, "cached_input": 0.40},
    "gpt-5.5":       {"input": 5.00,  "output": 30.00, "cached_input": 0.50},
}


def cost(response) -> float:
    model = response.model.split("-202")[0]
    p = PRICING.get(model, {})
    if not p:
        return 0.0
    u = response.usage
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)
    return (
        ((u.input_tokens - cached) / 1_000_000) * p["input"]
        + (cached / 1_000_000) * p["cached_input"]
        + (u.output_tokens / 1_000_000) * p["output"]
    )


# === Example 1: Basic usage ===
print("=" * 60)
print("EXAMPLE 1: GPT-6 Astra — basic call")
print("=" * 60)
print()

HARD_PROBLEM = (
    "A fintech startup is building a real-time fraud detection system that must: "
    "(1) evaluate every card transaction within 50ms, "
    "(2) handle 50K TPS at peak, "
    "(3) maintain <0.1% false-positive rate, "
    "(4) integrate with a legacy mainframe issuer system via batch files. "
    "Design a full production architecture. Cover: ingestion, feature engineering, "
    "model serving, feedback loops, regulatory auditability, and the mainframe integration "
    "strategy. Call out the three decisions that will most affect long-term maintainability."
)

response = client.responses.create(
    model="gpt-6-astra",
    input=HARD_PROBLEM,
)

print(f"Response ({response.usage.output_tokens} output tokens):\n")
print(response.output_text)
print(f"\nCost: ${cost(response):.5f}")


# === Example 2: Cross-model quality comparison on a hard coding task ===
print()
print("=" * 60)
print("EXAMPLE 2: Frontier comparison — Astra vs 5.6-Sol vs 5.5")
print("=" * 60)
print()

import time

CODING_TASK = (
    "Write a Python function `merge_interval_streams(streams)` that merges multiple "
    "sorted streams of (start, end) intervals — where streams can be infinite generators — "
    "into a single sorted, non-overlapping stream of merged intervals. "
    "The function should be memory-efficient (O(k) where k = number of streams), "
    "correct, and include a docstring with complexity analysis and three doctests."
)

results = []
for model in ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra"]:
    t0 = time.time()
    r = client.responses.create(model=model, input=CODING_TASK)
    elapsed = time.time() - t0
    c = cost(r)
    results.append({"model": model, "elapsed": elapsed, "cost": c,
                    "tokens": r.usage.output_tokens, "text": r.output_text})
    print(f"\n{'='*40}")
    print(f"MODEL: {model}")
    print(f"Latency: {elapsed:.2f}s | Output tokens: {r.usage.output_tokens} | Cost: ${c:.5f}")
    print(f"\n{r.output_text[:800]}...")

print("\n--- Summary ---")
print(f"{'Model':<18} {'Latency':>9} {'Out tok':>8} {'Cost':>12}")
for r in results:
    print(f"{r['model']:<18} {r['elapsed']:>8.2f}s {r['tokens']:>8} ${r['cost']:>10.5f}")


# === Example 3: GPT-6 Astra with reasoning effort ===
print()
print("=" * 60)
print("EXAMPLE 3: Reasoning effort on GPT-6 Astra")
print("=" * 60)
print()
print("GPT-6 Astra supports reasoning effort: low / medium / high / xhigh / max")
print()

LOGIC_PUZZLE = (
    "Five researchers (Alice, Bob, Carol, Dave, Eve) each use a different AI model "
    "(GPT-4.1, GPT-5.4, GPT-5.5, GPT-5.6-Sol, GPT-6 Astra). "
    "Clues: (1) The GPT-6 Astra user is adjacent to the GPT-5.5 user in alphabetical order. "
    "(2) Carol uses a model with a higher version than Bob's. "
    "(3) Dave uses GPT-5.4. "
    "(4) Alice's model has a lower version than Eve's. "
    "(5) The GPT-5.6-Sol user's name comes after 'C' alphabetically. "
    "Determine each researcher's model."
)

for effort in ["low", "high"]:
    r = client.responses.create(
        model="gpt-6-astra",
        input=LOGIC_PUZZLE,
        reasoning={"effort": effort},
    )
    reasoning_tokens = getattr(getattr(r.usage, "output_tokens_details", None), "reasoning_tokens", 0)
    print(f"Effort={effort:<6}: {r.usage.output_tokens:>5} output tokens "
          f"({reasoning_tokens} reasoning), cost=${cost(r):.5f}")
    print(f"  Answer: {r.output_text[:300].strip()}...")
    print()


# === Summary ===
print("=" * 60)
print("GPT-6 ASTRA QUICK REFERENCE")
print("=" * 60)
print("""
Model ID:  gpt-6-astra
Released:  September 3–4, 2026

Pricing (Sep 2026):
  Standard: $10.00 in / $50.00 out per 1M tokens
  Cached:   $1.00 / 1M (10% of input — standard cache discount)
  Batch:    $5.00 / $25.00 (50% off)
  Fast:     $20.00 / $100.00 (2x — lower latency path)

Context:   1M tokens

Reasoning effort: low / medium / high / xhigh / max
  Note: cyber-sensitive capabilities are gated behind a trusted-access program.

When to use:
  Use GPT-6 Astra for tasks where GPT-5.6 Sol still falls short:
    - Multi-step agentic research with strict accuracy requirements
    - Complex reasoning over large, dense documents
    - High-stakes code generation / formal verification
  For most production workloads, GPT-5.6 Terra or Luna remains more cost-effective.

Price positioning (Sep 2026):
  gpt-5.6-luna:  $0.20 in / $1.20 out
  gpt-5.6-terra: $2.00 in / $12.00 out
  gpt-5.6-sol:   $4.00 in / $20.00 out   (promo thru Nov 21)
  gpt-5.5:       $5.00 in / $30.00 out
  gpt-6-astra:   $10.00 in / $50.00 out  (2.5x Sol promo / 2x 5.5)
""")

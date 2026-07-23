"""Exercise 18: Cost and token tracking from Responses API."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Pricing per 1M tokens (verified July 23, 2026)
# Cached input READS follow the standard 10% rule for all families.
# GPT-5.6 additionally bills cache WRITES at 1.25x the input rate (new behavior).
# GPT-5.5 long-context: sessions >272K input tokens are billed at 2x input
# ($10.00/1M) and 1.5x output ($45.00/1M) for the ENTIRE session.
PRICING = {
    # GPT-5.6 family (July 9, 2026 GA) — Sol/Terra/Luna naming replaces base/mini/nano.
    # Cache writes: 1.25x input price. Cache reads: 10% input (same 10% rule as others).
    "gpt-5.6-sol":   {"input": 5.00,  "output": 30.00, "cached_input": 0.50},
    "gpt-5.6-terra": {"input": 2.50,  "output": 15.00, "cached_input": 0.25},
    "gpt-5.6-luna":  {"input": 1.00,  "output": 6.00,  "cached_input": 0.10},
    # GPT-5.5 (April 23, 2026 flagship) — largely superseded by 5.6-sol/terra.
    # Standard pricing applies only to sessions with <=272K input tokens.
    "gpt-5.5": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00, "cached_input": 3.00},
    # GPT-5.4 family (March 2026)
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 0.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25, "cached_input": 0.02},
    # GPT-4.1 family (cost-effective workhorse)
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    # Dated snapshots
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
}


def calculate_cost(response):
    """Extract usage and calculate cost from a Responses API response."""
    usage = response.usage
    model = response.model

    # Get pricing (try exact model name, then strip date suffix)
    prices = PRICING.get(model, PRICING.get(model.split("-202")[0], {}))
    if not prices:
        return {"error": f"No pricing for {model}"}

    # Token breakdown
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cached_tokens = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    non_cached_input = input_tokens - cached_tokens

    # Cost calculation
    input_cost = (non_cached_input / 1_000_000) * prices["input"]
    cached_cost = (cached_tokens / 1_000_000) * prices.get("cached_input", prices["input"])
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + cached_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "non_cached_input": non_cached_input,
        "output_tokens": output_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def print_cost_report(cost):
    """Pretty-print a cost breakdown."""
    print(f"  Model:            {cost['model']}")
    print(f"  Input tokens:     {cost['input_tokens']:>8} (cached: {cost['cached_tokens']})")
    print(f"  Output tokens:    {cost['output_tokens']:>8}")
    print(f"  Total tokens:     {cost['total_tokens']:>8}")
    print(f"  Input cost:       ${cost['input_cost']:.6f}")
    print(f"  Cached cost:      ${cost['cached_cost']:.6f}")
    print(f"  Output cost:      ${cost['output_cost']:.6f}")
    print(f"  TOTAL COST:       ${cost['total_cost']:.6f}")


# --- Run a few calls and track costs ---
print("=" * 60)
print("COST TRACKING DEMO")
print("=" * 60)

costs = []

# Call 1: Simple question
print("\n--- Call 1: Simple question (gpt-4.1-mini) ---")
r1 = client.responses.create(
    model="gpt-4.1-mini",
    input="What is the Responses API?",
)
c1 = calculate_cost(r1)
costs.append(c1)
print_cost_report(c1)

# Call 2: Longer generation
print("\n--- Call 2: Longer generation (gpt-4.1-mini) ---")
r2 = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a detailed 5-step implementation plan for deploying a RAG system in an enterprise environment.",
)
c2 = calculate_cost(r2)
costs.append(c2)
print_cost_report(c2)

# Call 3: With tools (higher token count)
print("\n--- Call 3: With web_search tool (gpt-4.1-mini) ---")
r3 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "web_search"}],
    input="What are the latest updates to OpenAI's enterprise offerings?",
)
c3 = calculate_cost(r3)
costs.append(c3)
print_cost_report(c3)

# Call 4: Premium model
print("\n--- Call 4: Same question on gpt-4.1 ---")
r4 = client.responses.create(
    model="gpt-4.1",
    input="Write a detailed 5-step implementation plan for deploying a RAG system in an enterprise environment.",
)
c4 = calculate_cost(r4)
costs.append(c4)
print_cost_report(c4)

# Call 5: Mid-tier reasoning model
print("\n--- Call 5: Same question on gpt-5.4-mini ---")
r5 = client.responses.create(
    model="gpt-5.4-mini",
    input="Write a detailed 5-step implementation plan for deploying a RAG system in an enterprise environment.",
)
c5 = calculate_cost(r5)
costs.append(c5)
print_cost_report(c5)

# Call 6: Current flagship — pricier per token but often more token-efficient
print("\n--- Call 6: Same question on gpt-5.5 (April 23, 2026 flagship) ---")
r6 = client.responses.create(
    model="gpt-5.5",
    input="Write a detailed 5-step implementation plan for deploying a RAG system in an enterprise environment.",
)
c6 = calculate_cost(r6)
costs.append(c6)
print_cost_report(c6)

# --- Summary ---
print("\n" + "=" * 60)
print("SESSION COST SUMMARY")
print("=" * 60)
total_session = sum(c["total_cost"] for c in costs)
total_tokens = sum(c["total_tokens"] for c in costs)

print(f"\n{'Call':<35} {'Tokens':>8} {'Cost':>12}")
print("-" * 60)
for i, c in enumerate(costs, 1):
    print(f"Call {i} ({c['model'][:20]}){'':<10} {c['total_tokens']:>8} ${c['total_cost']:>10.6f}")
print("-" * 60)
print(f"{'TOTAL':<35} {total_tokens:>8} ${total_session:>10.6f}")

print(f"\n--- Extrapolation ---")
calls_per_day = 10000
daily_cost = (total_session / len(costs)) * calls_per_day
monthly_cost = daily_cost * 30
print(f"Avg cost per call:  ${total_session / len(costs):.6f}")
print(f"At {calls_per_day:,} calls/day: ${daily_cost:.2f}/day, ${monthly_cost:.2f}/month")

print(f"\n--- Same prompt across model tiers ---")
print(f"  4.1-mini:  {c2['total_tokens']:>6} tokens, ${c2['total_cost']:.6f}")
print(f"  4.1:       {c4['total_tokens']:>6} tokens, ${c4['total_cost']:.6f}")
print(f"  5.4-mini:  {c5['total_tokens']:>6} tokens, ${c5['total_cost']:.6f}")
print(f"  5.5:       {c6['total_tokens']:>6} tokens, ${c6['total_cost']:.6f}")
if c2['total_cost'] > 0:
    print(f"\n  4.1 is        {c4['total_cost']/c2['total_cost']:.1f}x the cost of 4.1-mini")
    print(f"  5.4-mini is   {c5['total_cost']/c2['total_cost']:.1f}x the cost of 4.1-mini")
    print(f"  5.5 is        {c6['total_cost']/c2['total_cost']:.1f}x the cost of 4.1-mini")
print()
print("Watch token *count* not just per-token price — 5.6-sol is often cheaper")
print("end-to-end than 5.5 because it produces more concise output.")
print()
print("GPT-5.6 caching gotcha: cache WRITES are billed at 1.25x the input rate.")
print("  sol:   $6.25/1M to write; $0.50/1M to read back.")
print("  terra: $3.125/1M to write; $0.25/1M to read back.")
print("  luna:  $1.25/1M to write; $0.10/1M to read back.")
print("calculate_cost() tracks cache read cost. Cache write cost must be tracked")
print("separately if you use explicit prompt cache breakpoints (new in GPT-5.6).")
print()
print("GPT-5.5 caching gotcha: only EXTENDED prompt caching is supported.")
print("In-memory caching is unsupported — your cached_tokens will be 0 unless")
print("you've set up the extended prompt caching path (see prompt-caching docs).")
print()
print("GPT-5.5 long-context gotcha: sessions with >272K input tokens are billed")
print("at 2x input ($10.00/1M) and 1.5x output ($45.00/1M) for the FULL session.")
print("The calculate_cost() above uses standard rates — add a check if you send")
print("large contexts to avoid underestimating costs by 2x on input.")

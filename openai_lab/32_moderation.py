"""Exercise 32: Moderation scores in the Responses API (June 2026).

Pass a top-level `moderation` object in any generation request to receive
category scores for both the model input AND generated output — in the same
API call, without a separate /moderations round-trip.

Available in Responses API and Chat Completions API.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: Safe request — scores should be low across all categories ---
print("=" * 60)
print("EXAMPLE 1: Safe request with moderation enabled")
print("=" * 60)
print()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Explain the concept of retrieval-augmented generation in 3 sentences.",
    moderation={},
)

print(f"Output: {response.output_text[:300]}")
print()

# Moderation results live on response.moderation
mod = response.moderation
if mod:
    print("--- Input moderation ---")
    if mod.input and mod.input.results:
        for r in mod.input.results:
            print(f"  flagged: {r.flagged}")
            if r.category_scores:
                scores = vars(r.category_scores)
                top = sorted(scores.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)
                print(f"  top categories: {top[:3]}")

    print("\n--- Output moderation ---")
    if mod.output and mod.output.results:
        for r in mod.output.results:
            print(f"  flagged: {r.flagged}")
            if r.category_scores:
                scores = vars(r.category_scores)
                top = sorted(scores.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)
                print(f"  top categories: {top[:3]}")
else:
    print("(No moderation object in response — check that your account supports this feature)")

# --- Example 2: Use moderation for application-level routing ---
print()
print("=" * 60)
print("EXAMPLE 2: Application-level routing on moderation score")
print("=" * 60)
print()

SAFE_THRESHOLD = 0.3

test_inputs = [
    "How do I implement a binary search tree in Python?",
    "Write a poem about autumn leaves falling.",
]

for user_input in test_inputs:
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=user_input,
        moderation={},
    )

    flagged = False
    if r.moderation and r.moderation.input and r.moderation.input.results:
        flagged = any(res.flagged for res in r.moderation.input.results)

    status = "BLOCKED" if flagged else "OK"
    print(f"[{status}] '{user_input[:60]}'")
    if not flagged:
        print(f"  → {r.output_text[:120]}")
    print()

# --- Example 3: Inspect per-category scores for logging / audit trail ---
print()
print("=" * 60)
print("EXAMPLE 3: Per-category score inspection for audit logging")
print("=" * 60)
print()

r3 = client.responses.create(
    model="gpt-4.1-mini",
    input="Describe the historical causes of World War I.",
    moderation={},
)

print(f"Response preview: {r3.output_text[:200]}")
print()

if r3.moderation and r3.moderation.output and r3.moderation.output.results:
    result = r3.moderation.output.results[0]
    print("Output category scores (sorted by score):")
    if result.category_scores:
        scores = {k: v for k, v in vars(result.category_scores).items() if isinstance(v, float)}
        for category, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 30)
            print(f"  {category:<30} {score:.4f} {bar}")

print()
print("=" * 60)
print("MODERATION IN RESPONSES API — KEY POINTS")
print("=" * 60)
print("""
How to enable:
  response = client.responses.create(
      model="gpt-4.1-mini",
      input="...",
      moderation={},          # empty dict enables default moderation
  )

Where results live:
  response.moderation.input.results   — scores for the user's input
  response.moderation.output.results  — scores for the generated output

Fields per result:
  result.flagged           — True if any category exceeds its threshold
  result.categories        — bool per category (above threshold?)
  result.category_scores   — float 0.0–1.0 per category

Categories include: hate, harassment, self-harm, sexual, violence, and subtypes.

Use cases:
  1. Single-pass safety: replace a separate POST /moderations call
  2. Logging: store scores alongside generations for audit trails
  3. Routing: block, queue for review, or allow based on flagged + scores

Available in: Responses API and Chat Completions API (June 2026)
""")

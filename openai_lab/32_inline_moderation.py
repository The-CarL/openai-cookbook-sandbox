"""Exercise 32: Inline moderation in the Responses API.

Added May 2026. Pass moderation={"model": "omni-moderation-latest"} to
responses.create() and the API returns safety scores for both the model
input and the generated output inside the same response object — no
separate moderation call needed.

Why it matters: the standalone client.moderations.create() requires a
second round-trip and operates on a fixed text string. Inline moderation
runs concurrently with generation and scores the actual output the model
produced, which is the thing you'd actually act on.

Reference: OpenAI changelog, May 2026.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def print_moderation(label, mod_result):
    """Print a moderation result block."""
    if mod_result is None:
        print(f"  {label}: not available")
        return
    flagged = getattr(mod_result, "flagged", None)
    print(f"  {label}: flagged={flagged}")
    categories = getattr(mod_result, "categories", None)
    if categories:
        # Print any category that is True
        triggered = [k for k, v in vars(categories).items() if v is True]
        if triggered:
            print(f"    triggered categories: {', '.join(triggered)}")
        else:
            print(f"    no categories triggered")
    scores = getattr(mod_result, "category_scores", None)
    if scores:
        # Show top 3 scores
        score_dict = {k: v for k, v in vars(scores).items() if v is not None}
        top3 = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    top scores: " + ", ".join(f"{k}={v:.4f}" for k, v in top3))


# --- Example 1: Benign request (expected: not flagged) ---
print("=" * 60)
print("EXAMPLE 1: Benign input — expect clean scores")
print("=" * 60)
print()

response1 = client.responses.create(
    model="gpt-4.1-mini",
    input="Explain what a vector database is in one paragraph.",
    moderation={"model": "omni-moderation-latest"},
)

print(f"Output: {response1.output_text[:200]}")
print()
print("Moderation results:")
mod1 = getattr(response1, "moderation", None)
if mod1:
    print_moderation("input", getattr(mod1, "input", None))
    print_moderation("output", getattr(mod1, "output", None))
else:
    print("  (no moderation field — model may not support inline moderation)")

print(f"\nTokens: {response1.usage.input_tokens} in, {response1.usage.output_tokens} out")

# --- Example 2: Potentially sensitive content ---
print()
print("=" * 60)
print("EXAMPLE 2: Sensitive topic — show score gradient")
print("=" * 60)
print()

response2 = client.responses.create(
    model="gpt-4.1-mini",
    input=(
        "Write a brief fictional scene involving two characters in a heated argument. "
        "Keep it short — two exchanges max."
    ),
    moderation={"model": "omni-moderation-latest"},
)

print(f"Output: {response2.output_text[:300]}")
print()
print("Moderation results:")
mod2 = getattr(response2, "moderation", None)
if mod2:
    print_moderation("input", getattr(mod2, "input", None))
    print_moderation("output", getattr(mod2, "output", None))

print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")

# --- Example 3: Gate pattern — check before processing ---
print()
print("=" * 60)
print("EXAMPLE 3: Gate pattern — act on moderation result")
print("=" * 60)
print()

USER_INPUTS = [
    "Summarize the main ideas in the book 'The Lean Startup'.",
    "Write a story about a puppy finding a new home.",
    "How does photosynthesis work?",
]

for user_input in USER_INPUTS:
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=user_input,
        moderation={"model": "omni-moderation-latest"},
    )
    mod = getattr(r, "moderation", None)
    input_flagged = getattr(getattr(mod, "input", None), "flagged", False) if mod else False
    output_flagged = getattr(getattr(mod, "output", None), "flagged", False) if mod else False

    status = "FLAGGED" if (input_flagged or output_flagged) else "CLEAN"
    print(f"[{status}] '{user_input[:50]}...'")
    if input_flagged:
        print("         -> input was flagged, would block before processing")
    if output_flagged:
        print("         -> output was flagged, would suppress before returning to user")

# --- Summary ---
print()
print("=" * 60)
print("INLINE MODERATION KEY CONCEPTS")
print("=" * 60)
print("""
API call:
  response = client.responses.create(
      model="gpt-4.1-mini",
      input="...",
      moderation={"model": "omni-moderation-latest"},
  )

Result fields:
  response.moderation.input   — scores for the user's input
  response.moderation.output  — scores for the model's output
  Each has: .flagged (bool), .categories (per-category bool), .category_scores

When to use inline vs standalone:
  inline     — you need both generation and safety scores; saves a round-trip
  standalone — you want to moderate text you already have without generating

Gate pattern (typical production use):
  1. Call responses.create() with moderation={"model": "omni-moderation-latest"}
  2. Check response.moderation.input.flagged before doing downstream work
  3. Check response.moderation.output.flagged before returning to the user
  4. Log categories + category_scores for audit trails and policy tuning

Streaming note:
  Moderation scores arrive after the full output is complete — not
  interleaved with output deltas. Buffer the response before acting.

Treat flagged as a signal, not an automatic block:
  A model refusal that discusses harmful content can still trigger a flag.
  Calibrate thresholds against category_scores for your use case.
""")

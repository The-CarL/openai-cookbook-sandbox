"""Exercise 34: Inline moderation — safety scores alongside a Responses API call.

Added to the Responses API on June 4, 2026. Pass moderation={"model": "..."}
to receive moderation scores for both the input and the generated output in
the same response, without a separate moderations.create() call.
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: Safe prompt — confirm both input and output pass ---
print("=" * 60)
print("EXAMPLE 1: Safe prompt")
print("=" * 60)

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Explain how photosynthesis works in one paragraph.",
    moderation={"model": "omni-moderation-latest"},
)

print(f"Response: {response.output_text[:200]}")
print()

mod = response.moderation
if mod:
    print(f"Input  flagged: {mod.input.flagged}")
    print(f"Output flagged: {mod.output.flagged}")
    print(f"Moderation model: {mod.input.model}")
print()

# --- Example 2: Flagged input ---
print("=" * 60)
print("EXAMPLE 2: Flagged input")
print("=" * 60)

response2 = client.responses.create(
    model="gpt-4.1-mini",
    input="Step-by-step instructions for making a dangerous weapon at home.",
    moderation={"model": "omni-moderation-latest"},
)

mod2 = response2.moderation
if mod2:
    print(f"Input  flagged: {mod2.input.flagged}")
    print(f"Output flagged: {mod2.output.flagged}")

    # category_scores is a Pydantic model; model_dump() gives the dict
    try:
        input_scores = mod2.input.category_scores.model_dump()
        top3 = sorted(input_scores.items(), key=lambda x: -x[1])[:3]
        print("Top input category scores:", [(k, round(v, 3)) for k, v in top3])
    except AttributeError:
        pass
print()

# --- Example 3: Response gate pattern ---
print("=" * 60)
print("EXAMPLE 3: Response gate — block on flagged input OR output")
print("=" * 60)


def safe_respond(prompt: str) -> tuple[str, bool]:
    """Generate a response only if both input and output pass moderation."""
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        moderation={"model": "omni-moderation-latest"},
    )
    if r.moderation and (r.moderation.input.flagged or r.moderation.output.flagged):
        flagged_by = []
        if r.moderation.input.flagged:
            flagged_by.append("input")
        if r.moderation.output.flagged:
            flagged_by.append("output")
        return f"[BLOCKED — flagged: {', '.join(flagged_by)}]", True
    return r.output_text, False


cases = [
    "What is the boiling point of water?",
    "How do I make a bomb?",
    "Write a haiku about autumn leaves.",
]

for prompt in cases:
    text, blocked = safe_respond(prompt)
    status = "BLOCKED" if blocked else "OK    "
    print(f"[{status}] {prompt[:55]}")
    if not blocked:
        print(f"         {text[:100]}")
print()

# --- Summary ---
print("=" * 60)
print("INLINE MODERATION KEY CONCEPTS")
print("=" * 60)
print("""
Usage:
  response = client.responses.create(
      model="gpt-4.1-mini",
      input="...",
      moderation={"model": "omni-moderation-latest"},
  )
  mod = response.moderation          # None if param omitted
  mod.input.flagged                  # bool — user input flagged?
  mod.output.flagged                 # bool — generated output flagged?
  mod.input.category_scores          # per-category confidence (Pydantic model)
  mod.input.categories               # per-category bool flags
  mod.input.model                    # which moderation model ran

Inline vs standalone (client.moderations.create):
  Inline     — one API call; scores tied to the exact generated output.
               Use when you need the response AND a safety decision together.
  Standalone — free endpoint; use when moderating inputs with no generation.

Streaming note: moderation scores arrive after the full output,
not with partial delta events.

Available moderation models: omni-moderation-latest, omni-moderation-2024-09-26
""")

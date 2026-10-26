"""Exercise 39: Decisions API — typed answers, GA Oct 22 2026.

POST /v1/decisions evaluates text, images, and JSON and returns typed answers
about 10x faster than the Responses API. Powered by gpt-6-luna (the only
supported model). Uncached input is $0.10/M tokens; output tokens are free.

Question types:
  predicate — probability (0-1) that a condition is true
  choice    — one value from a fixed list, plus per-option probabilities
  score     — probability-weighted average of ordered level indices

Changes since the Oct 6 public beta (GA, Oct 22):
  - Prompt caching on shared `input` (implicit breakpoint, min 1,024 tokens)
  - Images inside individual question instructions
  - `input_json` content part for structured data
  - Up to 400 questions per request (was 200)

Use Decisions for classification, routing, and prioritisation. Use Structured
Outputs when you need generated fields or explanations.

Reference: https://developers.openai.com/api/docs/guides/decisions
Changelog: https://developers.openai.com/api/docs/changelog (Oct 6, Oct 22)
"""

import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
URL = "https://api.openai.com/v1/decisions"


def decide(body: dict) -> dict:
    """POST to /v1/decisions with plain httpx (works without SDK support)."""
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# --- Example 1: Multiple question types over shared input ---
print("=" * 60)
print("EXAMPLE 1: predicate + choice + score in one request")
print("=" * 60)

ticket = "I was charged twice for order 1182 and the app crashes when I try to see my invoice."

result = decide({
    "model": "gpt-6-luna",
    "input": ticket,
    "questions": [
        {
            "type": "predicate",
            "name": "duplicate_charge",
            "instructions": "Does the customer report being charged more than once?",
        },
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which team should handle this ticket?",
            "choices": ["billing", "engineering", "sales", "other"],
        },
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent is this ticket?",
            "levels": [
                {"description": "Informational, no action needed"},
                {"description": "Minor issue, can wait"},
                {"description": "Money or access affected, respond today"},
            ],
        },
    ],
})
for a in result.get("answers", []):
    print(f"  {a['name']}: {json.dumps({k: v for k, v in a.items() if k != 'name'})}")
print()

# --- Example 2: Structured data via input_json ---
print("=" * 60)
print("EXAMPLE 2: input_json (GA feature)")
print("=" * 60)

result = decide({
    "model": "gpt-6-luna",
    "input": [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Evaluate the ticket using the order evidence."},
            {"type": "input_json", "data": {
                "ticket": {"body": "Please refund the duplicate charge."},
                "order": {"charges": [49, 49]},
            }},
        ],
    }],
    "questions": [{
        "type": "predicate",
        "name": "duplicate_charge",
        "instructions": "Does the ticket describe a duplicate charge supported by the order?",
    }],
})
print(json.dumps(result.get("answers"), indent=2))
print()

# --- Example 3: Prompt caching across requests ---
print("=" * 60)
print("EXAMPLE 3: Shared input is cached; vary only the questions")
print("=" * 60)

# The implicit breakpoint sits at the end of `input`; the prefix needs >= 1,024 tokens.
policy = "Refund policy: " + ("Refunds are issued within 30 days of purchase. " * 200)
for q in ["Is the customer eligible for a refund?", "Does the policy mention a time limit?"]:
    result = decide({
        "model": "gpt-6-luna",
        "input": policy,
        "prompt_cache_key": "refund-policy-v1",
        "questions": [{"type": "predicate", "name": "q", "instructions": q}],
    })
    details = result.get("usage", {}).get("input_tokens_details", {})
    print(f"  cached_tokens={details.get('cached_tokens')} cache_write_tokens={details.get('cache_write_tokens')}")
print()
print("Tip: act on probabilities with thresholds tuned on labelled examples.")

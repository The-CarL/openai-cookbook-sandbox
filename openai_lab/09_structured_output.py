"""Exercise 9: Strict mode structured output using text.format."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Define a realistic enterprise schema: Customer Health Assessment
schema = {
    "type": "json_schema",
    "name": "customer_health_assessment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string"},
            "assessment_date": {"type": "string", "description": "ISO date format"},
            "health_score": {
                "type": "number",
                "description": "0-100 score based on overall health signals",
            },
            "risk_level": {
                "type": "string",
                "enum": ["healthy", "monitor", "at_risk", "critical"],
            },
            "usage_trend": {
                "type": "string",
                "enum": ["increasing", "stable", "declining"],
            },
            "key_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"],
                        },
                        "detail": {"type": "string"},
                    },
                    "required": ["signal", "sentiment", "detail"],
                    "additionalProperties": False,
                },
            },
            "recommended_actions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "next_review_date": {"type": "string"},
        },
        "required": [
            "customer_name",
            "assessment_date",
            "health_score",
            "risk_level",
            "usage_trend",
            "key_signals",
            "recommended_actions",
            "next_review_date",
        ],
        "additionalProperties": False,
    },
}

response = client.responses.create(
    model="gpt-4.1-mini",
    input="""Based on this account context, generate a customer health assessment:

Customer: Meridian Financial Services
- Contract: $480K/year, renewed 6 months ago
- API usage down 23% month-over-month for the last 3 months
- 2 P1 support tickets in the last 30 days (auth failures on their SSO integration)
- Champion contact (VP of Engineering) left the company 2 weeks ago
- Last QBR was 4 months ago, they cancelled the most recent one
- They recently started evaluating a competitor (found in their public RFP)
""",
    text={"format": schema},
)

# Parse and display the structured output
result = json.loads(response.output_text)
print("=== Customer Health Assessment (Structured Output) ===\n")
print(json.dumps(result, indent=2))

# Show that the output strictly matches our schema
print("\n=== Schema Compliance Check ===")
expected_fields = schema["schema"]["required"]
actual_fields = list(result.keys())
print(f"Expected fields: {expected_fields}")
print(f"Actual fields:   {actual_fields}")
print(f"All required fields present: {all(f in result for f in expected_fields)}")
print(f"Risk level valid enum: {result['risk_level'] in ['healthy', 'monitor', 'at_risk', 'critical']}")
print(f"Usage trend valid enum: {result['usage_trend'] in ['increasing', 'stable', 'declining']}")

# Verify signals have correct structure
for sig in result["key_signals"]:
    assert set(sig.keys()) == {"signal", "sentiment", "detail"}, f"Bad signal: {sig}"
    assert sig["sentiment"] in ["positive", "neutral", "negative"], f"Bad sentiment: {sig['sentiment']}"
print(f"All {len(result['key_signals'])} signals have valid structure and sentiment enums")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

"""Exercise 13: Parallel function calls — model requests multiple functions at once."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# Same backend as Exercise 12 (simplified)
def handle_function_call(name, args):
    data = {
        "get_revenue": {
            "Q1": {"revenue": 2_400_000, "growth": 12.3},
            "Q2": {"revenue": 2_800_000, "growth": 16.7},
            "Q3": {"revenue": 3_100_000, "growth": 10.7},
            "Q4": {"revenue": 3_500_000, "growth": 12.9},
        },
        "get_churn_rate": {
            "Q1": {"churn_rate": 2.1, "accounts_lost": 3},
            "Q2": {"churn_rate": 1.8, "accounts_lost": 2},
            "Q3": {"churn_rate": 3.5, "accounts_lost": 5},
            "Q4": {"churn_rate": 2.9, "accounts_lost": 4},
        },
        "get_nps_score": {
            "Q1": {"nps": 62, "responses": 145},
            "Q2": {"nps": 58, "responses": 132},
            "Q3": {"nps": 51, "responses": 167},
            "Q4": {"nps": 55, "responses": 155},
        },
        "get_support_volume": {
            "Q1": {"total_tickets": 234, "p1_tickets": 8, "avg_resolution_hours": 4.2},
            "Q2": {"total_tickets": 287, "p1_tickets": 12, "avg_resolution_hours": 5.1},
            "Q3": {"total_tickets": 356, "p1_tickets": 19, "avg_resolution_hours": 6.8},
            "Q4": {"total_tickets": 312, "p1_tickets": 15, "avg_resolution_hours": 5.5},
        },
    }
    quarter = args.get("quarter", "Q4")
    if name in data:
        return json.dumps(data[name].get(quarter, {"error": "Quarter not found"}))
    return json.dumps({"error": f"Unknown function: {name}"})


tools = [
    {
        "type": "function",
        "name": "get_revenue",
        "description": "Get revenue data for a specific quarter",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]}},
            "required": ["quarter"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_churn_rate",
        "description": "Get customer churn rate for a specific quarter",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]}},
            "required": ["quarter"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_nps_score",
        "description": "Get Net Promoter Score for a specific quarter",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]}},
            "required": ["quarter"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_support_volume",
        "description": "Get support ticket volume and metrics for a specific quarter",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]}},
            "required": ["quarter"],
            "additionalProperties": False,
        },
    },
]

# Craft a prompt that should trigger parallel calls
print("=== Parallel function calls ===\n")
response = client.responses.create(
    model="gpt-4.1-mini",
    tools=tools,
    input=(
        "Give me a complete Q3 business review. I need revenue, churn rate, "
        "NPS score, and support volume all for Q3. Fetch all metrics at once."
    ),
)

# Show that all calls were made in parallel (same response)
function_calls = [item for item in response.output if item.type == "function_call"]
print(f"Number of parallel function calls: {len(function_calls)}")
for fc in function_calls:
    print(f"  -> {fc.name}({fc.arguments})")

# Execute all
results_input = []
for fc in function_calls:
    args = json.loads(fc.arguments)
    result = handle_function_call(fc.name, args)
    results_input.append({
        "type": "function_call_output",
        "call_id": fc.call_id,
        "output": result,
    })
    print(f"  Result: {result}")

# Get final response
final = client.responses.create(
    model="gpt-4.1-mini",
    tools=tools,
    previous_response_id=response.id,
    input=results_input,
)

print(f"\n=== Q3 Business Review ===\n")
print(final.output_text)
print(f"\n--- Only 2 API calls needed (1 for parallel tool calls + 1 for synthesis) ---")

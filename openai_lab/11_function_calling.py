"""Exercise 11: Custom function calling — simulated CRM/enterprise backend."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Simulated enterprise backend ---
CRM_DATA = {
    "ACME-001": {
        "name": "Acme Corporation",
        "tier": "Enterprise",
        "contract_value": 480000,
        "renewal_date": "2026-09-15",
        "health_score": 72,
        "csm": "Sarah Chen",
    },
    "GLOB-002": {
        "name": "Globex International",
        "tier": "Professional",
        "contract_value": 120000,
        "renewal_date": "2026-06-01",
        "health_score": 45,
        "csm": "Marcus Rivera",
    },
    "INIT-003": {
        "name": "Initech Solutions",
        "tier": "Enterprise",
        "contract_value": 750000,
        "renewal_date": "2027-01-15",
        "health_score": 91,
        "csm": "Sarah Chen",
    },
}

USAGE_DATA = {
    "ACME-001": {"api_calls_30d": 1_250_000, "api_calls_prev_30d": 1_800_000, "active_users": 47},
    "GLOB-002": {"api_calls_30d": 340_000, "api_calls_prev_30d": 380_000, "active_users": 12},
    "INIT-003": {"api_calls_30d": 3_200_000, "api_calls_prev_30d": 2_900_000, "active_users": 83},
}


def handle_function_call(name, args):
    """Simulate backend function execution."""
    if name == "get_customer_details":
        customer_id = args["customer_id"]
        if customer_id in CRM_DATA:
            return json.dumps(CRM_DATA[customer_id])
        return json.dumps({"error": f"Customer {customer_id} not found"})
    elif name == "get_usage_metrics":
        customer_id = args["customer_id"]
        if customer_id in USAGE_DATA:
            return json.dumps(USAGE_DATA[customer_id])
        return json.dumps({"error": f"No usage data for {customer_id}"})
    elif name == "list_customers":
        return json.dumps([
            {"id": k, "name": v["name"], "tier": v["tier"]}
            for k, v in CRM_DATA.items()
        ])
    return json.dumps({"error": f"Unknown function: {name}"})


# --- Function definitions ---
tools = [
    {
        "type": "function",
        "name": "get_customer_details",
        "description": "Get detailed information about a customer from the CRM",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID (e.g., ACME-001)",
                },
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_usage_metrics",
        "description": "Get API usage metrics for a customer for the last 30 days",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer ID",
                },
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_customers",
        "description": "List all customers in the CRM",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]

# --- Make the initial call ---
print("=== Step 1: Initial request ===\n")
response = client.responses.create(
    model="gpt-4.1-mini",
    tools=tools,
    input="What's the current status of Acme Corporation? Include their contract details and recent usage trends.",
)

# Process function calls
print("Output items:")
for i, item in enumerate(response.output):
    print(f"  [{i}] type={item.type}", end="")
    if item.type == "function_call":
        print(f" -> {item.name}({item.arguments})")
    else:
        print()

# --- Step 2: Execute function calls and send results back ---
print("\n=== Step 2: Execute function calls ===\n")
function_results = []
for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = handle_function_call(item.name, args)
        print(f"Called {item.name}({args}) -> {result[:80]}...")
        function_results.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result,
        })

# --- Step 3: Send function results back to get the final answer ---
print("\n=== Step 3: Final response with function results ===\n")
final_response = client.responses.create(
    model="gpt-4.1-mini",
    tools=tools,
    previous_response_id=response.id,
    input=function_results,
)

print(final_response.output_text)
print(f"\n--- Total tokens: {response.usage.total_tokens + final_response.usage.total_tokens} ---")

"""Exercise 12: Multi-step agentic loop — complex question requiring multiple function calls."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Simulated enterprise backend (expanded) ---
CRM_DATA = {
    "ACME-001": {"name": "Acme Corporation", "tier": "Enterprise", "contract_value": 480000, "csm": "Sarah Chen"},
    "GLOB-002": {"name": "Globex International", "tier": "Professional", "contract_value": 120000, "csm": "Marcus Rivera"},
    "INIT-003": {"name": "Initech Solutions", "tier": "Enterprise", "contract_value": 750000, "csm": "Sarah Chen"},
    "WAYN-004": {"name": "Wayne Industries", "tier": "Enterprise", "contract_value": 920000, "csm": "Priya Patel"},
    "UMBR-005": {"name": "Umbrella Corp", "tier": "Professional", "contract_value": 200000, "csm": "Marcus Rivera"},
}

USAGE_DATA = {
    "ACME-001": {"api_calls_30d": 1_250_000, "api_calls_prev_30d": 1_800_000, "change_pct": -30.6},
    "GLOB-002": {"api_calls_30d": 340_000, "api_calls_prev_30d": 380_000, "change_pct": -10.5},
    "INIT-003": {"api_calls_30d": 3_200_000, "api_calls_prev_30d": 2_900_000, "change_pct": 10.3},
    "WAYN-004": {"api_calls_30d": 5_600_000, "api_calls_prev_30d": 2_100_000, "change_pct": 166.7},
    "UMBR-005": {"api_calls_30d": 890_000, "api_calls_prev_30d": 750_000, "change_pct": 18.7},
}

SUPPORT_TICKETS = {
    "ACME-001": [
        {"id": "TK-1001", "severity": "P2", "subject": "Sync delays on Salesforce connector", "status": "open"},
    ],
    "GLOB-002": [],
    "INIT-003": [
        {"id": "TK-1042", "severity": "P3", "subject": "Dashboard loading slowly", "status": "open"},
    ],
    "WAYN-004": [
        {"id": "TK-1055", "severity": "P1", "subject": "Data loss during migration batch", "status": "open"},
        {"id": "TK-1056", "severity": "P1", "subject": "Authentication failures after SSO migration", "status": "open"},
        {"id": "TK-1060", "severity": "P2", "subject": "Rate limiting on bulk API endpoint", "status": "open"},
    ],
    "UMBR-005": [
        {"id": "TK-1070", "severity": "P2", "subject": "Webhook delivery failures", "status": "open"},
    ],
}


def handle_function_call(name, args):
    if name == "list_customers":
        return json.dumps([{"id": k, "name": v["name"], "tier": v["tier"]} for k, v in CRM_DATA.items()])
    elif name == "get_customer_details":
        data = CRM_DATA.get(args["customer_id"])
        return json.dumps(data if data else {"error": "Not found"})
    elif name == "get_usage_metrics":
        data = USAGE_DATA.get(args["customer_id"])
        return json.dumps(data if data else {"error": "Not found"})
    elif name == "get_support_tickets":
        tickets = SUPPORT_TICKETS.get(args["customer_id"], [])
        open_tickets = [t for t in tickets if t["status"] == "open"]
        return json.dumps({"customer_id": args["customer_id"], "open_tickets": open_tickets})
    return json.dumps({"error": f"Unknown function: {name}"})


tools = [
    {
        "type": "function",
        "name": "list_customers",
        "description": "List all enterprise customers",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "get_customer_details",
        "description": "Get CRM details for a specific customer",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_usage_metrics",
        "description": "Get API usage metrics (last 30 days vs previous 30 days) for a customer",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_support_tickets",
        "description": "Get open support tickets for a customer",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
]

# --- THE AGENTIC LOOP ---
print("=== Starting agentic loop ===\n")
print('User: "Which of our enterprise customers had the biggest usage spike last week,')
print('       and do they have any open support tickets?"\n')

response = client.responses.create(
    model="gpt-4.1-mini",
    tools=tools,
    input=(
        "Which of our enterprise customers had the biggest usage spike recently, "
        "and do they have any open support tickets? "
        "Start by listing all customers, then check usage for each, "
        "find the biggest spike, and check their support tickets."
    ),
)

turn = 1
max_turns = 10  # Safety limit

while turn <= max_turns:
    # Check if the model wants to call functions
    function_calls = [item for item in response.output if item.type == "function_call"]

    if not function_calls:
        # No more function calls — model has its final answer
        break

    print(f"--- Turn {turn} ---")
    for fc in function_calls:
        args = json.loads(fc.arguments)
        print(f"  Call: {fc.name}({json.dumps(args)})")

    # Execute all function calls
    results_input = []
    for fc in function_calls:
        args = json.loads(fc.arguments)
        result = handle_function_call(fc.name, args)
        results_input.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": result,
        })
        print(f"  Result: {result[:100]}...")

    # Send results back
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=tools,
        previous_response_id=response.id,
        input=results_input,
    )
    turn += 1

print(f"\n=== Final answer (after {turn - 1} tool-call turns) ===\n")
print(response.output_text)

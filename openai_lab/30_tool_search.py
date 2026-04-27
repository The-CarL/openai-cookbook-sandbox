"""Exercise 30: Tool search — defer large tool surfaces until runtime.

When an agent has hundreds of tools, putting all of them in every request
is wasteful — most tokens are spent loading definitions the model will
never call. Tool search (Mar 5, 2026) lets you:

  1. Group tools into `namespace`s (e.g. "crm", "billing", "support").
  2. Mark individual tools `defer_loading: true` so their schemas only
     load when the model actually wants to use them.
  3. Add `{"type": "tool_search"}` so the model can search the deferred
     surface by name / description before calling.

Net effect: input tokens drop dramatically on requests that use only a
few tools out of a large catalog. This is the standard pattern for
production agents that wrap a real internal API surface.

Models: gpt-5.4 and later.

Reference: https://developers.openai.com/api/docs/guides/tools
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# === A realistic large tool surface ===
# Imagine these are 30+ tools backing a CSM platform. Most requests need 1-2.

def make_tool(name, description):
    """Build a function tool that's deferrable by default."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "defer_loading": True,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    }


tools = [
    # CRM namespace
    {
        "type": "namespace",
        "name": "crm",
        "description": "Customer record management — accounts, contacts, contracts.",
        "tools": [
            make_tool("get_account", "Fetch the full account record for a customer."),
            make_tool("list_contacts", "List all known contacts at the customer."),
            make_tool("get_contract", "Fetch contract terms and renewal date."),
            make_tool("get_health_score", "Fetch the latest computed health score."),
            make_tool("get_csm", "Fetch the assigned CSM for a customer."),
        ],
    },
    # Billing namespace
    {
        "type": "namespace",
        "name": "billing",
        "description": "Billing, invoicing, and payment status.",
        "tools": [
            make_tool("get_open_invoices", "List unpaid invoices for a customer."),
            make_tool("get_payment_history", "Fetch the last 12 months of payments."),
            make_tool("get_credit_balance", "Fetch current credit balance."),
            make_tool("get_subscription_status", "Fetch active subscriptions."),
        ],
    },
    # Support namespace
    {
        "type": "namespace",
        "name": "support",
        "description": "Support tickets, SLAs, and escalations.",
        "tools": [
            make_tool("list_open_tickets", "List all open support tickets."),
            make_tool("list_p1_tickets", "List only P1 (critical) open tickets."),
            make_tool("get_sla_status", "Get SLA compliance for the last 30 days."),
            make_tool("get_escalation_history", "List escalations in the last 90 days."),
        ],
    },
    # Usage namespace
    {
        "type": "namespace",
        "name": "usage",
        "description": "API and product usage analytics.",
        "tools": [
            make_tool("get_api_calls_30d", "API call volume in the last 30 days."),
            make_tool("get_active_users_30d", "Distinct active users in the last 30 days."),
            make_tool("get_feature_adoption", "Feature adoption percentages."),
            make_tool("get_data_volume", "Data volume processed in the last 30 days."),
        ],
    },
    # The discovery tool — model uses this to search the deferred surface
    {"type": "tool_search"},
]

total_tools = sum(len(ns["tools"]) for ns in tools if ns["type"] == "namespace")
print(f"=== Registered {total_tools} tools across {len(tools) - 1} namespaces ===")
print("All function tools have defer_loading=True; tool_search is enabled.\n")

# === Make a request that only needs 1-2 tools ===

print("=" * 70)
print("REQUEST: 'Is ACME-001 paying their invoices on time?'")
print("=" * 70)
print("Expectation: model uses tool_search to find billing tools, then calls one.\n")

response = client.responses.create(
    model="gpt-5.5",
    tools=tools,
    input="Is customer ACME-001 paying their invoices on time? Just check open invoices.",
    parallel_tool_calls=False,
)

# Walk the output to see what the model actually loaded
print("Output items:")
for i, item in enumerate(response.output):
    if item.type == "tool_search_call":
        print(f"  [{i}] tool_search_call (model is browsing the catalog)")
    elif item.type == "function_call":
        print(f"  [{i}] function_call -> {item.name}({item.arguments})")
    elif item.type == "message":
        text = ""
        for c in item.content:
            if c.type == "output_text":
                text = c.text[:120]
        print(f"  [{i}] message: {text}...")
    else:
        print(f"  [{i}] {item.type}")

print(f"\nInput tokens: {response.usage.input_tokens}")
print("(Compare to a baseline that loads all 18 tool schemas upfront — this")
print(" should be substantially lower because most schemas were deferred.)")

# === When to use tool search ===

print("\n" + "=" * 70)
print("WHEN TO USE TOOL SEARCH")
print("=" * 70)
print("""
Use tool search when:
  - You have 20+ tools and individual requests use a small subset
  - Your tool schemas are large (rich descriptions, deeply nested params)
  - You're paying for cached input but the cache misses because the tool
    list shifts request-to-request
  - You want to expose a vendor's full API surface without bloating every call

Don't use tool search when:
  - You have <10 tools (overhead isn't worth it)
  - Most requests use most tools (defer_loading hurts)
  - The model needs to reason about the *full* tool surface to plan
    (e.g. multi-tool composition decisions)

Implementation tips:
  - Group by *user intent*, not by API surface. "billing", "support", and
    "usage" are good namespaces; "rest_endpoints_v2" is not.
  - Write tool descriptions like search-engine snippets — the model uses
    them to decide what to load.
  - Set parallel_tool_calls=False when using tool_search; the model often
    needs to search-then-call sequentially.
  - Pairs naturally with prompt caching (Exercise 25): keep your namespace
    structure stable so the cached prefix stays valid.
""")

"""Exercise 35: Programmatic Tool Calling — GPT-5.6 coordinates tools via hosted JavaScript.

GA with the GPT-5.6 family (July 9, 2026). Available on gpt-5.6-sol, gpt-5.6-terra,
gpt-5.6-luna and the bare alias gpt-5.6 (routes to Sol).

How it works
────────────
Regular tool use:  model → function_call → you execute → function_call_output → model
                   N tools = N model round-trips.

Programmatic TC:   model writes JavaScript → V8 runtime calls your functions → model
                   N tools = 1 model call + N function executions in the runtime.

The model generates a JavaScript program that runs in an isolated, stateless V8 runtime
managed by OpenAI (no Node.js, no network, no filesystem). That program coordinates your
registered functions — in parallel, with loops, with conditionals — and returns a final
result without repeatedly asking the model what to do next.

Efficiency gains cited by OpenAI: 38–64% fewer total tokens and 50% fewer model turns
compared to standard tool use on the same tasks.

Enabling it
───────────
1. Add {"type": "programmatic_tool_calling"} to your tools list.
2. Add "allowed_callers": ["programmatic"] to each function tool the program can invoke.
   Omit "allowed_callers" (or set to ["direct"]) to keep a tool reserved for direct calls.

Response shape
──────────────
output may contain:
  program          — the generated JS code + call_id
  function_call    — a tool invocation FROM the program (has caller.caller_id == program call_id)
  message          — the final text answer

When you return function results for programmatic calls, copy the caller field from the
function_call item so the runtime knows which program to resume.

Reference: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# ---- Simulated back-end functions -------------------------------------------

def get_product_info(sku: str) -> dict:
    catalog = {
        "sku_001": {"name": "Widget Pro", "price": 49.99, "category": "hardware"},
        "sku_002": {"name": "Gadget Lite", "price": 19.99, "category": "accessories"},
        "sku_003": {"name": "Thingamajig", "price": 129.00, "category": "hardware"},
    }
    return catalog.get(sku, {"error": f"SKU {sku} not found"})


def get_inventory(sku: str) -> dict:
    stock = {
        "sku_001": {"available_units": 42, "warehouse": "US-WEST"},
        "sku_002": {"available_units": 0,  "warehouse": "US-EAST"},
        "sku_003": {"available_units": 7,  "warehouse": "US-WEST"},
    }
    return stock.get(sku, {"error": f"No inventory record for {sku}"})


def get_recent_reviews(sku: str) -> dict:
    reviews = {
        "sku_001": {"count": 238, "avg_rating": 4.6, "top_complaint": "packaging"},
        "sku_002": {"count": 55,  "avg_rating": 3.9, "top_complaint": "battery life"},
        "sku_003": {"count": 12,  "avg_rating": 4.8, "top_complaint": "none notable"},
    }
    return reviews.get(sku, {"error": f"No reviews for {sku}"})


FUNCTION_MAP = {
    "get_product_info":    get_product_info,
    "get_inventory":       get_inventory,
    "get_recent_reviews":  get_recent_reviews,
}

# ---- Tool definitions --------------------------------------------------------

TOOLS = [
    # Hosted runtime that lets the model write coordinating JS
    {"type": "programmatic_tool_calling"},
    # Function tools available to the program
    {
        "type": "function",
        "name": "get_product_info",
        "description": "Return name, price, and category for a product SKU.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "Product SKU code"}},
            "required": ["sku"],
        },
        "allowed_callers": ["programmatic"],
    },
    {
        "type": "function",
        "name": "get_inventory",
        "description": "Return stock levels and warehouse location for a SKU.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "Product SKU code"}},
            "required": ["sku"],
        },
        "allowed_callers": ["programmatic"],
    },
    {
        "type": "function",
        "name": "get_recent_reviews",
        "description": "Return review count, average rating, and top complaint for a SKU.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "Product SKU code"}},
            "required": ["sku"],
        },
        "allowed_callers": ["programmatic"],
    },
]


# ---- The agentic loop --------------------------------------------------------

def run_programmatic_tool_call(query: str) -> str:
    """Run one request with programmatic tool calling, handling all function calls."""
    import json

    print(f"Query: {query}\n")

    response = client.responses.create(
        model="gpt-5.6-sol",
        tools=TOOLS,
        input=query,
    )

    # Process output items until we get a message with no outstanding programs
    pending_calls = {}   # call_id → {"type": function_name, caller: ...}
    function_results = []

    for item in response.output:
        print(f"  [{item.type}]", end="")

        if item.type == "program":
            # The model emitted JS code — note it but we don't execute it;
            # OpenAI's runtime runs it and will emit function_call items for us to handle.
            print(f"  call_id={item.call_id}")
            print(f"  --- generated program (excerpt) ---")
            code_preview = item.code[:300] if hasattr(item, "code") else "(code not in SDK field)"
            print(f"  {code_preview}...")

        elif item.type == "function_call":
            fname = item.name
            args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            print(f"  name={fname} args={args}")

            # Execute the function locally
            result = FUNCTION_MAP[fname](**args)
            print(f"    → {result}")

            # Queue the result, preserving caller so the runtime can resume
            fc_result = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result),
            }
            # Copy caller field if present (links this result back to the program)
            if hasattr(item, "caller") and item.caller:
                fc_result["caller"] = item.caller
            function_results.append(fc_result)

        elif item.type == "message":
            print()
            final_text = "".join(
                c.text for c in item.content if hasattr(c, "text")
            )
            if final_text:
                print(f"\nAnswer:\n{final_text}")
                return final_text

    # If there were programmatic function calls, send all results back in one shot
    if function_results:
        print(f"\n  → Returning {len(function_results)} function result(s) to runtime...")
        response2 = client.responses.create(
            model="gpt-5.6-sol",
            tools=TOOLS,
            previous_response_id=response.id,
            input=function_results,
        )
        for item in response2.output:
            if item.type == "message":
                final_text = "".join(
                    c.text for c in item.content if hasattr(c, "text")
                )
                if final_text:
                    print(f"\nAnswer:\n{final_text}")
                    return final_text

    return "(no text output)"


# ---- Examples ----------------------------------------------------------------

print("=" * 60)
print("EXAMPLE 1: Three SKUs — product info + inventory + reviews")
print("=" * 60)
print("""
Without Programmatic Tool Calling, answering this would require multiple
model round-trips: one per (SKU × data type) call sequence.
With it, the model writes JS that fetches all 9 data points concurrently
in the hosted V8 runtime, then summarises them in one final model call.
""")

run_programmatic_tool_call(
    "Give me a purchase-decision summary for SKUs sku_001, sku_002, and sku_003. "
    "For each: price, stock status, and whether customer ratings support buying."
)

print()
print("=" * 60)
print("PROGRAMMATIC TOOL CALLING KEY CONCEPTS")
print("=" * 60)
print("""
Enable:
  Add {"type": "programmatic_tool_calling"} to tools.
  Add "allowed_callers": ["programmatic"] to each function tool the program can call.

Tool access modes:
  "allowed_callers": ["direct"]        — model calls it the normal way (direct only)
  "allowed_callers": ["programmatic"]  — model can only call via JS program
  "allowed_callers": ["direct", "programmatic"] — either mode
  (omit allowed_callers)               — default: direct only, not usable from program

Response items:
  program        — the JS code OpenAI's runtime executes (you don't execute this)
  function_call  — a tool invocation FROM the program; has caller.caller_id
  message        — the final answer after the program completes

Returning results:
  Return function_call_output items as usual, but COPY the caller field from
  the function_call item — the runtime uses it to resume the right program.

When to use it:
  ✓ Multi-SKU lookups, parallel data fetching, loops over a list of items
  ✓ Tool-heavy pipelines where you'd otherwise write a custom agentic loop
  ✗ Tasks that need fresh model judgment between each tool call
  ✗ Zero-Data-Retention (ZDR) workloads where server-side execution is prohibited
      (check your data processing agreement before enabling)

Efficiency:
  OpenAI benchmarks: 38–64% fewer total tokens, ~50% fewer model turns vs direct calls.
  Works on gpt-5.6-luna, gpt-5.6-terra, gpt-5.6-sol (alias: gpt-5.6).
""")

"""Exercise 24: OpenAI Agents SDK — multi-agent workflows with handoffs and guardrails.

Requires: pip install openai-agents  (or: uv add openai-agents)
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel


# === Example 1: Basic agent with a function tool ===

@function_tool
def get_customer_details(customer_id: str) -> str:
    """Look up customer details from the CRM."""
    crm = {
        "ACME-001": "Acme Corp — Enterprise tier, $480K/yr, CSM: Sarah Chen, health: 72",
        "GLOB-002": "Globex Inc — Professional tier, $120K/yr, CSM: Marcus Rivera, health: 45",
        "INIT-003": "Initech — Enterprise tier, $750K/yr, CSM: Sarah Chen, health: 91",
    }
    return crm.get(customer_id, f"Customer {customer_id} not found")


@function_tool
def get_usage_metrics(customer_id: str) -> str:
    """Get API usage metrics for a customer."""
    usage = {
        "ACME-001": "1.25M calls/30d (down 30% MoM), 47 active users",
        "GLOB-002": "340K calls/30d (down 10% MoM), 12 active users",
        "INIT-003": "3.2M calls/30d (up 10% MoM), 83 active users",
    }
    return usage.get(customer_id, f"No usage data for {customer_id}")


csm_agent = Agent(
    name="Customer Success Agent",
    instructions=(
        "You are a customer success manager assistant. Use the available tools "
        "to look up customer information and provide actionable insights. "
        "Always check both CRM details and usage metrics before giving advice."
    ),
    tools=[get_customer_details, get_usage_metrics],
)


async def example_1():
    print("=" * 60)
    print("EXAMPLE 1: Basic agent with function tools")
    print("=" * 60)
    print()

    result = await Runner.run(
        csm_agent,
        "What's the status of Acme Corp (ACME-001)? Should I be worried?",
    )
    print(f"Final output:\n{result.final_output}")
    print()


# === Example 2: Multi-agent handoffs ===

technical_agent = Agent(
    name="Technical Support",
    handoff_description="Handles technical questions about API integration, latency, errors",
    instructions=(
        "You are a technical support specialist. Help with API issues, "
        "debugging, performance optimization, and integration questions."
    ),
    tools=[get_usage_metrics],
)

billing_agent = Agent(
    name="Billing Support",
    handoff_description="Handles billing, pricing, contract, and invoice questions",
    instructions=(
        "You are a billing specialist. Help with pricing tiers, contract terms, "
        "invoices, and payment questions."
    ),
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the front-line triage agent. Determine the nature of the "
        "customer's request and hand off to the appropriate specialist. "
        "Use Technical Support for API/integration issues and Billing Support "
        "for pricing/contract questions."
    ),
    handoffs=[technical_agent, billing_agent],
)


async def example_2():
    print("=" * 60)
    print("EXAMPLE 2: Multi-agent handoffs")
    print("=" * 60)
    print()

    # Technical question — should route to technical_agent
    print("--- Query: Technical question ---")
    result = await Runner.run(
        triage_agent,
        "Our API calls are timing out after we upgraded to the new SDK version. "
        "We're seeing 504 errors on about 10% of requests.",
    )
    print(f"Handled by: {result.last_agent.name}")
    print(f"Output: {result.final_output[:300]}...")
    print()

    # Billing question — should route to billing_agent
    print("--- Query: Billing question ---")
    result2 = await Runner.run(
        triage_agent,
        "We want to upgrade from Professional to Enterprise tier. "
        "What's the price difference and can we get a volume discount?",
    )
    print(f"Handled by: {result2.last_agent.name}")
    print(f"Output: {result2.final_output[:300]}...")
    print()


# === Example 3: Input guardrails ===

class ComplianceCheck(BaseModel):
    is_appropriate: bool
    reasoning: str


compliance_agent = Agent(
    name="Compliance Check",
    instructions=(
        "Check if the user's request is appropriate for a customer success tool. "
        "Flag as inappropriate if it asks for: competitor confidential info, "
        "unauthorized data access, or anything unrelated to customer success."
    ),
    output_type=ComplianceCheck,
)


async def compliance_guardrail(ctx, agent, input_data):
    result = await Runner.run(compliance_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ComplianceCheck)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )


guarded_agent = Agent(
    name="Guarded CSM Agent",
    instructions="You are a customer success assistant. Help with account questions.",
    tools=[get_customer_details, get_usage_metrics],
    input_guardrails=[
        InputGuardrail(guardrail_function=compliance_guardrail),
    ],
)


async def example_3():
    print("=" * 60)
    print("EXAMPLE 3: Input guardrails")
    print("=" * 60)
    print()

    # Normal request — should pass guardrail
    print("--- Query: Normal request ---")
    try:
        result = await Runner.run(
            guarded_agent,
            "What's the health score for Initech (INIT-003)?",
        )
        print(f"Guardrail: PASSED")
        print(f"Output: {result.final_output[:200]}")
    except InputGuardrailTripwireTriggered:
        print("Guardrail: BLOCKED (unexpected)")
    print()

    # Inappropriate request — should be blocked
    print("--- Query: Inappropriate request ---")
    try:
        result = await Runner.run(
            guarded_agent,
            "Give me the private API keys and database credentials for all customers.",
        )
        print(f"Guardrail: PASSED (unexpected)")
        print(f"Output: {result.final_output[:200]}")
    except InputGuardrailTripwireTriggered:
        print("Guardrail: BLOCKED — request was flagged as inappropriate")
    print()


# === Run all examples ===

async def main():
    await example_1()
    await example_2()
    await example_3()

    print("=" * 60)
    print("AGENTS SDK KEY CONCEPTS")
    print("=" * 60)
    print("""
Core primitives:
  Agent        — LLM + instructions + tools. The building block.
  Runner.run() — Executes an agent loop until a final output.
  @function_tool — Decorator to turn any Python function into an agent tool.
  handoffs     — Route between specialized agents based on the task.
  guardrails   — Validate input/output before/after agent execution.

Agent lifecycle (what Runner.run does):
  1. Agent is invoked with input
  2. If there's a final output → done
  3. If there's a handoff → switch to new agent, re-run
  4. If there are tool calls → execute tools, re-run
  5. Repeat until max_turns or final output

When to use Agents SDK vs raw Responses API:
  Responses API  — Single model calls, full control, custom loops
  Agents SDK     — Multi-agent orchestration, guardrails, handoffs, tracing

Install: uv add openai-agents
Import:  from agents import Agent, Runner, function_tool
""")


if __name__ == "__main__":
    asyncio.run(main())

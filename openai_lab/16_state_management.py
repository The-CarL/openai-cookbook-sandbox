"""Exercise 16: Conversation state management patterns for enterprise chatbots."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Pattern 1: previous_response_id (simple, stateless server) ---
print("=" * 60)
print("PATTERN 1: previous_response_id (API-managed state)")
print("=" * 60)
print()
print("Pros: Zero state management code. Just store the response ID.")
print("Cons: No control over context window. Can't edit/summarize history.")
print("Best for: Prototypes, simple chatbots, internal tools.")
print()

r1 = client.responses.create(
    model="gpt-4.1-mini",
    input="My name is Alex and I work at Meridian Financial. We're on the Enterprise tier.",
)
print(f"Turn 1 -> response_id: {r1.id}")

r2 = client.responses.create(
    model="gpt-4.1-mini",
    input="What tier did I say I was on?",
    previous_response_id=r1.id,
)
print(f"Turn 2 -> {r2.output_text}")
print(f"         (used previous_response_id={r1.id[:30]}...)")
print()
print("Storage needed: Just one ID per conversation (the latest response_id).")
print(f"That's it: '{r2.id}'")

# --- Pattern 2: Manual message array (database-backed) ---
print()
print("=" * 60)
print("PATTERN 2: Manual input array (app-managed state)")
print("=" * 60)
print()
print("Pros: Full control. Can summarize, edit, inject system context.")
print("Cons: You manage the message array, truncation, token counting.")
print("Best for: Production apps, multi-user systems, complex workflows.")
print()

# Simulate what you'd store in a database
conversation_history = [
    {"role": "user", "content": "My name is Alex and I work at Meridian Financial. We're on the Enterprise tier."},
]

r3 = client.responses.create(
    model="gpt-4.1-mini",
    input=conversation_history,
)

# Store assistant response in history
conversation_history.append({"role": "assistant", "content": r3.output_text})

# Next turn — you control exactly what goes in
conversation_history.append({"role": "user", "content": "What tier did I say I was on?"})

r4 = client.responses.create(
    model="gpt-4.1-mini",
    input=conversation_history,
)
print(f"Turn 2 -> {r4.output_text}")
print()
print(f"History stored ({len(conversation_history)} messages):")
for msg in conversation_history:
    print(f"  [{msg['role']}] {msg['content'][:60]}...")

# --- Pattern 3: Hybrid (previous_response_id + system context injection) ---
print()
print("=" * 60)
print("PATTERN 3: Hybrid (API state + system context)")
print("=" * 60)
print()
print("Pros: Simple state management + rich system context from your DB.")
print("Cons: System context uses tokens every turn.")
print("Best for: Enterprise chatbots with user-specific data.")
print()

# Pull customer data from your database
customer_context = """You are a customer success assistant for CloudSync.
Current customer context (from CRM):
- Customer: Meridian Financial Services
- Tier: Enterprise ($480K/year)
- Health Score: 72 (at risk)
- CSM: Sarah Chen
- Open tickets: 2 P1 tickets (SSO auth failures)
- Usage trend: Down 23% MoM for 3 months
- Champion left 2 weeks ago
"""

r5 = client.responses.create(
    model="gpt-4.1-mini",
    instructions=customer_context,
    input="What should I be most worried about with this account?",
)
print(f"Response: {r5.output_text}")

# Continue with previous_response_id — the instructions carry forward
r6 = client.responses.create(
    model="gpt-4.1-mini",
    input="Draft an email to their new VP of Engineering introducing myself as their CSM.",
    previous_response_id=r5.id,
)
print(f"\nFollow-up: {r6.output_text[:300]}...")

print()
print("=" * 60)
print("DECISION FRAMEWORK")
print("=" * 60)
print("""
When to use each pattern:

1. previous_response_id only:
   - Internal tools, demos, prototypes
   - Single-session conversations
   - When you don't need to persist across server restarts

2. Manual message array (DB-backed):
   - Multi-user production systems
   - When you need conversation history search/analytics
   - When you need to summarize old messages to manage context window
   - When conversations span multiple sessions or devices

3. Hybrid (previous_response_id + instructions):
   - Enterprise chatbots with CRM integration
   - When you want simple state management but rich per-user context
   - When conversation context should include real-time data from your systems
""")

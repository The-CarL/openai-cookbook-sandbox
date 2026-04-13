"""Exercise 2: Multi-turn conversation using previous_response_id."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Turn 1: Set the context
print("=" * 60)
print("TURN 1")
print("=" * 60)
r1 = client.responses.create(
    model="gpt-4.1-mini",
    input="I'm evaluating three LLM providers for our enterprise platform: OpenAI, Anthropic, and Google. We need strong function calling, structured outputs, and enterprise security. What should I consider?",
)
print(f"User: I'm evaluating three LLM providers...\n")
print(f"Assistant: {r1.output_text}\n")
print(f"Response ID: {r1.id}")
print(f"Tokens: {r1.usage.input_tokens} in, {r1.usage.output_tokens} out")

# Turn 2: Follow up — the model remembers Turn 1 via previous_response_id
print("\n" + "=" * 60)
print("TURN 2")
print("=" * 60)
r2 = client.responses.create(
    model="gpt-4.1-mini",
    input="Narrow it down. Which one has the best function calling reliability and structured output support?",
    previous_response_id=r1.id,  # <-- This is the magic. No message array needed.
)
print(f"User: Which one has the best function calling reliability?\n")
print(f"Assistant: {r2.output_text}\n")
print(f"Response ID: {r2.id}")
print(f"Previous ID: {r1.id}")
print(f"Tokens: {r2.usage.input_tokens} in, {r2.usage.output_tokens} out")

# Turn 3: Go deeper
print("\n" + "=" * 60)
print("TURN 3")
print("=" * 60)
r3 = client.responses.create(
    model="gpt-4.1-mini",
    input="OK, what about data residency and compliance? Our legal team needs SOC 2 Type II, HIPAA BAA, and EU data residency.",
    previous_response_id=r2.id,  # Chains to the full conversation
)
print(f"User: What about data residency and compliance?\n")
print(f"Assistant: {r3.output_text}\n")
print(f"Response ID: {r3.id}")
print(f"Previous ID: {r2.id}")
print(f"Tokens: {r3.usage.input_tokens} in, {r3.usage.output_tokens} out")

# Show how token counts grow as context accumulates
print("\n" + "=" * 60)
print("TOKEN GROWTH ACROSS TURNS")
print("=" * 60)
for i, r in enumerate([r1, r2, r3], 1):
    print(f"Turn {i}: {r.usage.input_tokens:>5} input, {r.usage.output_tokens:>5} output, {r.usage.total_tokens:>5} total")
print("\nNotice: input tokens grow each turn because the full conversation is included.")
print("But YOU never had to manage a messages array — the API does it via previous_response_id.")

"""Exercise 8: Combine web_search + code_interpreter in one call."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {"type": "web_search"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
    instructions="Use web_search to find data, then use code_interpreter to analyze it. Show your work.",
    input=(
        "Find the current market caps of NVIDIA, Apple, and Microsoft. "
        "Then use code_interpreter to calculate each company's percentage share "
        "of the combined total, and determine how much NVIDIA would need to grow "
        "to match Apple's market cap."
    ),
)

# Show the tool-use chain
print("=== Tool usage chain ===\n")
for i, item in enumerate(response.output):
    if item.type == "web_search_call":
        print(f"[{i}] WEB_SEARCH (id={item.id[:20]}...)")
    elif item.type == "code_interpreter_call":
        code_preview = item.code.strip().split("\n")[0]
        print(f"[{i}] CODE_INTERPRETER: {code_preview}...")
    elif item.type == "message":
        text = ""
        for c in item.content:
            if c.type == "output_text":
                text = c.text[:80]
        print(f"[{i}] MESSAGE: {text}...")

print("\n=== Full response ===\n")
# Collect all text from all message items
for item in response.output:
    if item.type == "message":
        for c in item.content:
            if c.type == "output_text":
                print(c.text)
                print()

print(f"--- Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out ---")
print(f"--- Output items: {len(response.output)} ---")

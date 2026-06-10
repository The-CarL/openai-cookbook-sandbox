"""Exercise 5: Web search tool in the Responses API."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "web_search"}],
    input="What were OpenAI's most significant product announcements in the last 30 days?",
)

print("=== Response ===\n")
print(response.output_text)

# Show all output items — web_search results appear as tool call items
print("\n=== Output items ===")
for i, item in enumerate(response.output):
    print(f"\n[{i}] type={item.type}")
    if item.type == "web_search_call":
        print(f"    id={item.id}")
        print(f"    status={item.status}")
    elif item.type == "message":
        for j, content in enumerate(item.content):
            if content.type == "output_text":
                # Show annotations (citations)
                if content.annotations:
                    print(f"    Citations found: {len(content.annotations)}")
                    for k, ann in enumerate(content.annotations):
                        print(f"    [{k}] {ann.type}: {ann.title}")
                        print(f"         URL: {ann.url}")

print(f"\n=== Usage ===")
print(f"Input:  {response.usage.input_tokens} tokens")
print(f"Output: {response.usage.output_tokens} tokens")

# --- return_token_budget: longer reasoning-backed web search (May 2026) ---
# Only applies to GPT-5+ reasoning models via the Responses API web_search tool.
# Default behaviour caps the number of tokens the model can consume from search
# results; "unlimited" removes that cap for high-effort research or eval runs.
# Use selectively — it increases latency and cost.
print("\n=== return_token_budget=unlimited (high-effort research) ===\n")
print("GPT-5+ reasoning models only. Removes the default search-result token cap.")
print("Useful for deep research tasks that need to read many pages.")
print()

response2 = client.responses.create(
    model="gpt-5.4-mini",
    tools=[{
        "type": "web_search",
        "return_token_budget": "unlimited",
    }],
    input=(
        "What are the three most significant OpenAI API changes released in "
        "May and June 2026? Be specific about model IDs and parameter names."
    ),
)

print(response2.output_text)
print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")
print()
print("Note: compare token counts vs. the default web_search call above.")
print("return_token_budget=unlimited typically produces higher token consumption")
print("because the model reads more search-result content before summarising.")

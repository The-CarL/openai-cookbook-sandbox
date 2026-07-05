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

# --- Example 2: include raw search results (SDK 2.33 / 2026-04-28) ---
# Pass include=["web_search_call.results"] to get the raw hits back in the
# web_search_call item — useful when you want to post-process sources yourself.
print("\n=== Example 2: include=['web_search_call.results'] ===\n")

response2 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "web_search"}],
    include=["web_search_call.results"],
    input="What OpenAI models were released in April 2026?",
)

print(response2.output_text)
print("\n=== Raw search hits from web_search_call.results ===")
for item in response2.output:
    if item.type == "web_search_call":
        results = getattr(item, "results", None)
        if results:
            print(f"  {len(results)} raw hit(s) returned:")
            for r in results[:5]:
                print(f"    - {getattr(r, 'title', '(no title)')}: {getattr(r, 'url', '')}")
        else:
            print("  (no results field — check that include=[\"web_search_call.results\"] is set)")

print(f"\nInput: {response2.usage.input_tokens} tokens | Output: {response2.usage.output_tokens} tokens")

# --- Example 3: return_token_budget — longer reasoning-backed search (May 2026) ---
# Only applies to GPT-5+ reasoning models. Removes the default cap on how many
# tokens the model can consume from search results. Use for high-effort research.
print("\n=== Example 3: return_token_budget='unlimited' (GPT-5+ only) ===\n")

response3 = client.responses.create(
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

print(response3.output_text)
print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")
print()
print("return_token_budget='unlimited' lets the model read more search-result")
print("content before summarising — higher quality at the cost of more tokens.")

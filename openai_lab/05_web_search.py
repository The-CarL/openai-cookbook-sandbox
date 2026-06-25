"""Exercise 5: Web search tool in the Responses API."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: Basic web search ---
print("=== Example 1: Basic web search ===\n")
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

# --- Example 2: Domain filtering ---
print("\n\n=== Example 2: Domain filtering ===\n")
response2 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{
        "type": "web_search",
        "allowed_domains": ["arxiv.org", "semanticscholar.org"],
    }],
    input="Summarize recent research on retrieval-augmented generation (RAG) performance benchmarks.",
)
print(response2.output_text)

# --- Example 3: return_token_budget (GPT-5+ reasoning web search, May 2026) ---
# return_token_budget controls how much search content the tool can return.
# Set "unlimited" for deep-research or eval runs that need to inspect many pages.
# Has no effect on non-reasoning models or web_search_preview.
print("\n\n=== Example 3: High-effort research with return_token_budget=unlimited ===\n")
response3 = client.responses.create(
    model="gpt-5.5",
    tools=[{
        "type": "web_search",
        "return_token_budget": "unlimited",
    }],
    input=(
        "Do a thorough comparison of the top three vector database solutions "
        "available today: feature set, pricing, managed vs self-hosted options, "
        "and which workloads each excels at. Cite sources."
    ),
)
print(response3.output_text)
print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")

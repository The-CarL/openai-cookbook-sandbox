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

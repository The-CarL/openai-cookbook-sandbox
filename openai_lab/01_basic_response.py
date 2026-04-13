"""Exercise 1: Basic Responses API call."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="What are the three most important things an AI Solutions Engineer should understand about LLM APIs? Be concise.",
)

# Show the simple way to get text
print("=== Quick access ===")
print(response.output_text)

# Show the full response structure
print("\n=== Response metadata ===")
print(f"ID:             {response.id}")
print(f"Model:          {response.model}")
print(f"Status:         {response.status}")
print(f"Usage (input):  {response.usage.input_tokens} tokens")
print(f"Usage (output): {response.usage.output_tokens} tokens")
print(f"Usage (total):  {response.usage.total_tokens} tokens")

# Show the output structure — it's a list of output items
print("\n=== Output structure ===")
for i, item in enumerate(response.output):
    print(f"Output[{i}]: type={item.type}")
    if item.type == "message":
        for j, content in enumerate(item.content):
            print(f"  Content[{j}]: type={content.type}, text={content.text[:80]}...")

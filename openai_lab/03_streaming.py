"""Exercise 3: Streaming with the Responses API."""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

print("=== Streaming response ===\n")

start = time.time()
first_token_time = None
event_types_seen = set()

stream = client.responses.create(
    model="gpt-4.1-mini",
    input="Explain the difference between RAG and fine-tuning for enterprise AI deployments. Be concise — 3 bullet points.",
    stream=True,
)

print("Text: ", end="", flush=True)
for event in stream:
    event_types_seen.add(type(event).__name__)

    # ResponseTextDeltaEvent has the streaming text chunks
    if event.type == "response.output_text.delta":
        if first_token_time is None:
            first_token_time = time.time()
        print(event.delta, end="", flush=True)

    # ResponseCompletedEvent has the final response with usage
    if event.type == "response.completed":
        final_response = event.response

end = time.time()

print("\n")
print("=" * 60)
print("STREAMING METRICS")
print("=" * 60)
print(f"Time to first token: {(first_token_time - start)*1000:.0f}ms")
print(f"Total time:          {(end - start)*1000:.0f}ms")
print(f"Tokens (input):      {final_response.usage.input_tokens}")
print(f"Tokens (output):     {final_response.usage.output_tokens}")

print(f"\nEvent types seen ({len(event_types_seen)}):")
for et in sorted(event_types_seen):
    print(f"  - {et}")

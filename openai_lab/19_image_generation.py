"""Exercise 19: Image generation tool in the Responses API."""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Basic image generation ---
print("=== Basic image generation ===\n")
response = client.responses.create(
    model="gpt-4.1-mini",
    input=(
        "Generate a clean, professional logo for a cloud data sync platform "
        "called CloudSync. Use blue and white colors, modern minimalist style."
    ),
    tools=[{"type": "image_generation"}],
)

# Extract image data from response
print("Output items:")
for i, item in enumerate(response.output):
    if item.type == "image_generation_call":
        print(f"  [{i}] IMAGE_GENERATION ({len(item.result)} base64 chars)")
        image_bytes = base64.b64decode(item.result)
        with open("cloudsync_logo.png", "wb") as f:
            f.write(image_bytes)
        print(f"       Saved to cloudsync_logo.png ({len(image_bytes)} bytes)")
    elif item.type == "message":
        for c in item.content:
            if c.type == "output_text":
                print(f"  [{i}] MESSAGE: {c.text[:120]}...")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# --- Image generation with options ---
print("\n=== Image generation with transparent background ===\n")
response2 = client.responses.create(
    model="gpt-4.1-mini",
    input="Create a simple icon of a database with sync arrows, flat design style",
    tools=[{
        "type": "image_generation",
        "background": "transparent",
        "quality": "high",
    }],
)

for i, item in enumerate(response2.output):
    if item.type == "image_generation_call":
        print(f"  [{i}] IMAGE_GENERATION (transparent, high quality)")
        image_bytes = base64.b64decode(item.result)
        with open("sync_icon.png", "wb") as f:
            f.write(image_bytes)
        print(f"       Saved to sync_icon.png ({len(image_bytes)} bytes)")

print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")

# --- Combined: text + image in one response ---
print("\n=== Multi-tool: web_search + image_generation ===\n")
response3 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[
        {"type": "web_search"},
        {"type": "image_generation"},
    ],
    input=(
        "What does the OpenAI logo look like? Describe it, then generate "
        "a similar minimalist AI company logo (not a copy — inspired by the style)."
    ),
)

print("Tool usage chain:")
for i, item in enumerate(response3.output):
    if item.type == "web_search_call":
        print(f"  [{i}] WEB_SEARCH")
    elif item.type == "image_generation_call":
        print(f"  [{i}] IMAGE_GENERATION")
        image_bytes = base64.b64decode(item.result)
        with open("ai_logo_inspired.png", "wb") as f:
            f.write(image_bytes)
        print(f"       Saved to ai_logo_inspired.png")
    elif item.type == "message":
        for c in item.content:
            if c.type == "output_text":
                print(f"  [{i}] MESSAGE: {c.text[:120]}...")

print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")

# --- Cleanup ---
print("\n=== Cleanup ===")
for f in ["cloudsync_logo.png", "sync_icon.png", "ai_logo_inspired.png"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

print("\n=== Key takeaways ===")
print("- image_generation is a built-in Responses API tool, like web_search")
print("- Output items have type='image_generation_call' with base64 result")
print("- Supports options: background='transparent', quality='high'")
print("- Can combine with other tools (web_search, code_interpreter) in one call")
print("- Works with gpt-4.1-mini, gpt-4.1, gpt-5.4 family, and others")

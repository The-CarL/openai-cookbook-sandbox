"""Exercise 32: gpt-image-2 — direct Images API: generation, editing, token pricing, Batch."""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# gpt-image-2 (API available to developers May 2026) is accessed via the Images API
# directly — NOT via the Responses API image_generation tool used in Exercise 19.
#
# Key differences from Exercise 19:
#   API endpoint:  client.images.generate(model="gpt-image-2", ...)
#   Pricing:       token-based (text + image tokens) — not a flat per-image fee
#   Editing:       client.images.edit(model="gpt-image-2", ...) in one round-trip
#   Batch API:     50% off all rates for async workloads
#   Quality:       higher fidelity for multilingual text, infographics, diagrams

# --- Overview ---
print("=" * 60)
print("GPT-IMAGE-2: Direct Images API")
print("=" * 60)
print()
print("gpt-image-2 generates and edits images via the Images API endpoint.")
print("Pricing is token-based: text input tokens + image output tokens.")
print("See platform.openai.com/docs/pricing for current per-token rates.")
print()

# --- Example 1: Basic image generation ---
print("=" * 60)
print("EXAMPLE 1: Generate an image")
print("=" * 60)
print()

response = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A clean technical diagram showing a three-tier web architecture: "
        "browser client, API server, and database. Use minimal flat design, "
        "blue and grey tones, clear labels on each tier."
    ),
    n=1,
    size="1024x1024",
    response_format="b64_json",
)

image_bytes = base64.b64decode(response.data[0].b64_json)
with open("arch_diagram.png", "wb") as f:
    f.write(image_bytes)
print(f"Generated: arch_diagram.png ({len(image_bytes):,} bytes)")

# Token usage — gpt-image-2 exposes token counts unlike DALL-E flat billing
if hasattr(response, "usage") and response.usage:
    u = response.usage
    print(f"Token usage — input: {getattr(u, 'input_tokens', '?')}, "
          f"output: {getattr(u, 'output_tokens', '?')}")

# --- Example 2: Image editing ---
print()
print("=" * 60)
print("EXAMPLE 2: Edit the generated image")
print("=" * 60)
print()
print("client.images.edit() sends the original image + a text instruction.")
print("The model returns a modified version in a single API call.")
print()

with open("arch_diagram.png", "rb") as img_file:
    edit_response = client.images.edit(
        model="gpt-image-2",
        image=img_file,
        prompt=(
            "Add a CDN / Edge layer above the browser client tier, "
            "with a cloud icon and an arrow connecting it to the browser."
        ),
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

edited_bytes = base64.b64decode(edit_response.data[0].b64_json)
with open("arch_diagram_edited.png", "wb") as f:
    f.write(edited_bytes)
print(f"Edited:    arch_diagram_edited.png ({len(edited_bytes):,} bytes)")

if hasattr(edit_response, "usage") and edit_response.usage:
    u = edit_response.usage
    print(f"Token usage — input: {getattr(u, 'input_tokens', '?')}, "
          f"output: {getattr(u, 'output_tokens', '?')}")

# --- Example 3: Compact icon generation (256×256 → fewer output tokens) ---
print()
print("=" * 60)
print("EXAMPLE 3: Small icon (256×256 — cheapest output token count)")
print("=" * 60)
print()

icon_response = client.images.generate(
    model="gpt-image-2",
    prompt="A simple green checkmark icon on a transparent background, flat design.",
    n=1,
    size="256x256",
    response_format="b64_json",
)

icon_bytes = base64.b64decode(icon_response.data[0].b64_json)
with open("checkmark.png", "wb") as f:
    f.write(icon_bytes)
print(f"Icon:      checkmark.png ({len(icon_bytes):,} bytes)")

if hasattr(icon_response, "usage") and icon_response.usage:
    u = icon_response.usage
    print(f"Token usage — input: {getattr(u, 'input_tokens', '?')}, "
          f"output: {getattr(u, 'output_tokens', '?')}")

# --- Cleanup ---
print()
print("=== Cleanup ===")
for path in ["arch_diagram.png", "arch_diagram_edited.png", "checkmark.png"]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed {path}")

# --- Summary ---
print()
print("=" * 60)
print("KEY CONCEPTS: gpt-image-2")
print("=" * 60)
print("""
Generation:
  client.images.generate(
      model="gpt-image-2",
      prompt="...",
      n=1,                       # number of images
      size="1024x1024",          # or "256x256", "512x512", "1792x1024", etc.
      response_format="b64_json" # or "url"
  )

Editing (key new capability vs. gpt-image-1):
  client.images.edit(
      model="gpt-image-2",
      image=open("image.png", "rb"),
      prompt="describe the change",
      n=1,
      size="1024x1024",
      response_format="b64_json",
  )

Token-based pricing (verify at platform.openai.com/docs/pricing):
  Text input tokens  — prompt text consumed
  Image input tokens — input image pixels (edit flow)
  Image output tokens — generated image pixels (larger size = more tokens = more cost)
  Batch API:          50% off via client.batches for async workloads

Size → cost guide:
  256×256   — icon / thumbnail   (fewest output tokens)
  512×512   — small
  1024×1024 — standard (most common)
  1792×1024 — landscape widescreen
  1024×1792 — portrait tall

vs. Responses API image_generation tool (Exercise 19):
  Exercise 19  — model orchestrates generation mid-conversation; no direct token visibility
  Exercise 32  — standalone generation/editing; explicit token usage; Batch API available

New in gpt-image-2 vs. gpt-image-1:
  - Single-call image editing (client.images.edit)
  - Higher fidelity for multilingual text, infographics, slides, diagrams
  - Token-based pricing instead of flat per-image fee
  - Batch API for 50% cost reduction on async workloads
""")

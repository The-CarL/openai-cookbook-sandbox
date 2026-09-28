"""Exercise 32: gpt-image-2 / gpt-image-2.5 — Images API: generation, editing, token pricing."""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# gpt-image-2 (API available to developers May 2026) is accessed via the Images API
# directly — NOT via the Responses API image_generation tool used in Exercise 19.
#
# gpt-image-2.5 released September 8, 2026 with two variants:
#   gpt-image-2.5-flare     — fast default; up to 50% lower latency than gpt-image-2
#   gpt-image-2.5-sunburst  — high-quality editing and premium creative work
#
# Pricing for gpt-image-2.5 (token-based, same for both variants):
#   Text input:        $5.00 / 1M tokens
#   Image input:       $8.00 / 1M tokens  (edit flow)
#   Image output:      $30.00 / 1M tokens
#   Cached text input: $1.25 / 1M tokens
#   Cached image input: $2.00 / 1M tokens
#   Per-image cost:    ~$0.006 (low quality) to ~$0.211 (max, 1024×1024)
#
# Key differences from Exercise 19:
#   API endpoint:  client.images.generate(model="gpt-image-2", ...)
#   Pricing:       token-based (text + image tokens) — not a flat per-image fee
#   Editing:       client.images.edit(model="gpt-image-2", ...) in one round-trip
#   Batch API:     50% off all rates for async workloads
#   Quality:       higher fidelity for multilingual text, infographics, diagrams

# --- Overview ---
print("=" * 60)
print("GPT-IMAGE-2 / GPT-IMAGE-2.5: Direct Images API")
print("=" * 60)
print()
print("gpt-image-2 and gpt-image-2.5 generate and edit images via the Images API endpoint.")
print("Pricing is token-based: text input + image output (and image input for edits).")
print("gpt-image-2.5 (Sep 8, 2026): Flare variant = fast; Sunburst = quality/editing.")
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
print("KEY CONCEPTS: gpt-image-2 / gpt-image-2.5")
print("=" * 60)
print("""
Generation:
  client.images.generate(
      model="gpt-image-2",          # or "gpt-image-2.5-flare" / "gpt-image-2.5-sunburst"
      prompt="...",
      n=1,                          # number of images
      size="1024x1024",             # or "256x256", "512x512", "1792x1024", etc.
      response_format="b64_json"    # or "url"
  )

Editing (key new capability vs. gpt-image-1):
  client.images.edit(
      model="gpt-image-2",          # or "gpt-image-2.5-sunburst" for best edit quality
      image=open("image.png", "rb"),
      prompt="describe the change",
      n=1,
      size="1024x1024",
      response_format="b64_json",
  )

Model selection (as of Sep 2026):
  gpt-image-2           — stable baseline; good quality, proven API
  gpt-image-2.5-flare   — faster; up to 50% lower latency than gpt-image-2; everyday generation
  gpt-image-2.5-sunburst — slower; best editing precision; premium creative work

Token-based pricing for gpt-image-2.5 ($/1M):
  Text input:         $5.00   Image input:  $8.00   Image output: $30.00
  Cached text input:  $1.25   Cached image: $2.00
  Batch API:          50% off via client.batches for async workloads

Size → cost guide:
  256×256   — icon / thumbnail   (fewest output tokens, ~$0.006/image low-q)
  512×512   — small
  1024×1024 — standard (most common, ~$0.211/image at max quality)
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

New in gpt-image-2.5 (Sep 8, 2026) vs. gpt-image-2:
  - Two variants: Flare (speed) and Sunburst (quality/editing)
  - Up to 50% lower latency on Flare
  - Sketch feature (ChatGPT UI): turn drawings into AI images
  - Same token-based pricing model; identical rates for both variants
""")

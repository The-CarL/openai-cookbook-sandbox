"""Exercise 32: gpt-image-2 — standalone Images API with native reasoning.

gpt-image-2 went live in the API on April 28, 2026 (ChatGPT launched April 21).
It is OpenAI's first image model with native reasoning: it reads the prompt,
reasons about composition and style, and then renders pixels — producing far
better text rendering and brand-consistent product shots than prior models.

Two endpoints are available:
  /v1/images/generations  — client.images.generate()   (text → image)
  /v1/images/edits        — client.images.edit()        (image + prompt → image)

Pricing is token-based (verify at openai.com/api/pricing):
  Text input tokens:    ~$5.00 / 1M
  Image input tokens:   ~$8.00 / 1M
  Image output tokens:  ~$30.00 / 1M
  Batch (async):        50% off all of the above

This is different from Exercise 19, which uses the `image_generation` built-in
tool inside the Responses API. Use THIS exercise when you need:
  - Direct control over size, quality, background
  - Image editing (modify an existing image)
  - Batch API pricing
Use Exercise 19 when you want image generation as one step in a larger
Responses API workflow (e.g., web_search → summarize → draw diagram).

Note: DALL-E 2 and DALL-E 3 reach end-of-life on May 12, 2026.
      Migrate to gpt-image-2 (or gpt-image-1.5) before that date.
"""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# --- Example 1: Basic generation ---
print("=" * 60)
print("EXAMPLE 1: Basic image generation with gpt-image-2")
print("=" * 60)
print()

response = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A clean, professional hero image for a cloud data platform. "
        "Dark background, glowing blue network nodes, modern minimalist style. "
        "Include the word 'CloudSync' in crisp white sans-serif text."
    ),
    size="1536x1024",   # landscape; options: 1024x1024, 1536x1024, 1024x1536, auto
    quality="standard", # standard or hd (hd ≈ 3–4x more output tokens)
    n=1,
)

image_data = response.data[0].b64_json  # gpt-image-2 always returns base64
image_bytes = base64.b64decode(image_data)
with open("cloudsync_hero.png", "wb") as f:
    f.write(image_bytes)
print(f"Generated: cloudsync_hero.png ({len(image_bytes):,} bytes)")
print(f"Revised prompt: {response.data[0].revised_prompt or '(none)'}")


# --- Example 2: HD quality + transparent background ---
print()
print("=" * 60)
print("EXAMPLE 2: HD quality with transparent background")
print("=" * 60)
print()

response2 = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A flat-design icon of a database with circular sync arrows. "
        "Blue and white. No background. Use clean geometric shapes."
    ),
    size="1024x1024",
    quality="hd",
    background="transparent",  # requires output_format png or webp
    n=1,
)

image_bytes2 = base64.b64decode(response2.data[0].b64_json)
with open("sync_icon_hd.png", "wb") as f:
    f.write(image_bytes2)
print(f"Generated (HD, transparent): sync_icon_hd.png ({len(image_bytes2):,} bytes)")


# --- Example 3: Image editing ---
print()
print("=" * 60)
print("EXAMPLE 3: Image editing — alter an existing image")
print("=" * 60)
print()

# Edit the hero image we just generated
with open("cloudsync_hero.png", "rb") as img_file:
    edit_response = client.images.edit(
        model="gpt-image-2",
        image=img_file,
        prompt=(
            "Keep the same layout and text. Replace the background with a deep "
            "purple gradient instead of dark blue. Keep the white network nodes."
        ),
        size="1536x1024",
    )

image_bytes3 = base64.b64decode(edit_response.data[0].b64_json)
with open("cloudsync_hero_purple.png", "wb") as f:
    f.write(image_bytes3)
print(f"Edited: cloudsync_hero_purple.png ({len(image_bytes3):,} bytes)")


# --- Cleanup ---
print()
print("=" * 60)
print("Cleanup")
print("=" * 60)
for fname in ["cloudsync_hero.png", "sync_icon_hd.png", "cloudsync_hero_purple.png"]:
    if os.path.exists(fname):
        os.remove(fname)
        print(f"Removed {fname}")


# --- Summary ---
print()
print("=" * 60)
print("GPT-IMAGE-2 KEY CONCEPTS")
print("=" * 60)
print("""
Endpoints:
  client.images.generate()  — text-to-image
  client.images.edit()      — image + prompt → image (optional mask for inpainting)

Parameters:
  model      = "gpt-image-2"
  size       = "1024x1024" | "1536x1024" (landscape) | "1024x1536" (portrait) | "auto"
  quality    = "standard" | "hd"           (hd ≈ 3-4x output tokens = 3-4x cost)
  background = "auto" | "transparent" | "opaque"
  n          = 1..10                       (up to 8 coherent images per prompt)
  Output is always b64_json — response.data[i].b64_json

Pricing (token-based — verify at openai.com/api/pricing):
  Text input:    ~$5/M tokens
  Image input:   ~$8/M tokens
  Image output:  ~$30/M tokens
  Batch API:     50% off — use client.batches.create() for async image jobs

gpt-image-2 vs image_generation Responses tool (Exercise 19):
  gpt-image-2 (this exercise):
    + Direct control: size, quality, background, editing, batch pricing
    + Better for asset pipelines, editing workflows, cost optimization
  image_generation tool (Exercise 19):
    + Seamlessly integrates into Responses API flows
    + Combine with web_search, code_interpreter, etc. in one call
    + Best when image is one step in a larger agent workflow

DALL-E deprecation: DALL-E 2 and DALL-E 3 shut down May 12, 2026.
Migrate to gpt-image-2 (generation) or gpt-image-1.5 (editing default).
""")

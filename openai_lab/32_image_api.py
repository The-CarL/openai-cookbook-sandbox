"""Exercise 32: gpt-image-2 — standalone image generation and editing API.

DALL-E 2 and DALL-E 3 were deprecated and removed from the API on May 12, 2026.
gpt-image-2 (released April 21, 2026) is the replacement. It supports
O-series reasoning before generating, text rendering in images, and flexible
resolutions up to 4K.

Contrast with Exercise 19: that exercise uses the built-in image_generation
tool inside responses.create(). This exercise calls client.images.generate()
directly — you control quality, size, and get raw image bytes back.

Requires: pip install openai python-dotenv
"""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# --- Example 1: Basic generation, low quality (fast + cheap) ---
print("=" * 60)
print("EXAMPLE 1: Basic generation — low quality, 1024×1024")
print("=" * 60)
print()

response = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A clean, professional product photo of a modern wireless keyboard "
        "on a white background, studio lighting, top-down view."
    ),
    size="1024x1024",
    quality="low",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(response.data[0].b64_json)
with open("keyboard_low.png", "wb") as f:
    f.write(image_bytes)
print(f"Saved keyboard_low.png ({len(image_bytes):,} bytes)")
print(f"Quality: low  | Est. cost: ~$0.006")

# --- Example 2: High quality, portrait orientation ---
print()
print("=" * 60)
print("EXAMPLE 2: High quality — 1024×1536 portrait")
print("=" * 60)
print()

response2 = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A minimalist book cover for 'The API Economy' — "
        "abstract network nodes in deep blue and gold, space for title at top, "
        "author name at bottom, white background."
    ),
    size="1024x1536",
    quality="high",
    n=1,
    response_format="b64_json",
)

image_bytes2 = base64.b64decode(response2.data[0].b64_json)
with open("book_cover_high.png", "wb") as f:
    f.write(image_bytes2)
print(f"Saved book_cover_high.png ({len(image_bytes2):,} bytes)")
print(f"Quality: high | Est. cost: ~$0.35")

# --- Example 3: Medium quality, landscape orientation ---
print()
print("=" * 60)
print("EXAMPLE 3: Medium quality — 1536×1024 landscape")
print("=" * 60)
print()

response3 = client.images.generate(
    model="gpt-image-2",
    prompt=(
        "A LinkedIn hero banner for a cloud computing startup — "
        "minimalist datacenter imagery, blue gradient, clean and professional. "
        "No text."
    ),
    size="1536x1024",
    quality="medium",
    n=1,
    response_format="b64_json",
)

image_bytes3 = base64.b64decode(response3.data[0].b64_json)
with open("banner_medium.png", "wb") as f:
    f.write(image_bytes3)
print(f"Saved banner_medium.png ({len(image_bytes3):,} bytes)")
print(f"Quality: medium | Est. cost: ~$0.075")

# --- Cleanup ---
print()
for f in ["keyboard_low.png", "book_cover_high.png", "banner_medium.png"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

# --- Reference: Image editing pattern ---
print()
print("=" * 60)
print("IMAGE EDITING PATTERN (reference — requires your own input image)")
print("=" * 60)
print("""
# Inpainting: replace a masked region with generated content.
# The mask PNG must be the same size as the image; transparent pixels
# mark the area to be replaced.

response = client.images.edit(
    model="gpt-image-2",
    image=open("photo.png", "rb"),
    mask=open("mask.png", "rb"),   # transparent pixels = area to edit
    prompt="Replace the background with a modern open-plan office.",
    size="1024x1024",
)
image_bytes = base64.b64decode(response.data[0].b64_json)
""")

# --- Key concepts ---
print("=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Migration from DALL-E (deprecated and removed May 12, 2026):
  OLD: client.images.generate(model="dall-e-3", quality="hd", ...)
  NEW: client.images.generate(model="gpt-image-2", quality="high", ...)
  Same client, same endpoint — just update model and quality vocab.

Quality tiers (gpt-image-2, 1024×1024 approx pricing, June 2026):
  "low"    ~$0.006/image  — fast, good for drafts and thumbnails
  "medium" ~$0.053/image  — balanced quality/cost for most use cases
  "high"   ~$0.211/image  — photorealistic detail, text-in-images

Size options:
  "1024x1024"  — square (default)
  "1024x1536"  — portrait
  "1536x1024"  — landscape
  Arbitrary WxH supported: both dims must be divisible by 16,
  aspect ratio between 1:3 and 3:1, max ~3840×2160.

Response format:
  response_format="b64_json"  — base64-encoded PNG bytes (no expiry)
  (default returns .url, which expires after ~1 hour)

vs. Exercise 19 (image_generation built-in tool):
  Ex 19: tools=[{"type": "image_generation"}] inside responses.create()
         → model decides when to generate; output is part of the response
  Ex 32: client.images.generate() → direct, synchronous, full control

Other replacement models for former DALL-E workloads:
  gpt-image-1      — general-purpose, also token-based pricing
  gpt-image-1-mini — cheaper, lower fidelity
  gpt-image-2      — flagship: best text rendering, reasoning, 4K support
""")

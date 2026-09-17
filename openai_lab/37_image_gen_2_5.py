"""Exercise 37: GPT-Image-2.5 Flare and Sunburst — September 2026 image models.

Released September 8, 2026 as a split-workload upgrade to gpt-image-2:

  gpt-image-2.5-flare    — Speed tier. High-quality everyday generation and editing.
                           Best for high-volume workflows, rapid prototyping, creator content.
  gpt-image-2.5-sunburst — Precision tier. Slower but with higher editing accuracy.
                           Best when exact inpainting, text rendering, or diagram fidelity matter.

Both expose the same Images API endpoints (generate + edit) with the same token pricing.
New in 2.5: xhigh and max quality settings (gpt-image-2 capped at high).

Pricing (September 2026) — token-based, same for both Flare and Sunburst:
  Image input tokens:  $8.00 / 1M
  Cached image input:  $2.00 / 1M
  Image output tokens: $30.00 / 1M
  Text input tokens:   $5.00 / 1M
  Cached text input:   $1.25 / 1M

Access via the Images API endpoint (not the Responses API image_generation tool).
Both support Batch API at 50% off for async generation workloads.
"""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

IMAGE_PRICING = {
    "image_input":        8.00,
    "cached_image_input": 2.00,
    "image_output":      30.00,
    "text_input":         5.00,
    "cached_text_input":  1.25,
}


def image_cost(response) -> float:
    u = getattr(response, "usage", None)
    if not u:
        return 0.0
    return (
        getattr(u, "input_tokens", 0) / 1_000_000 * IMAGE_PRICING["text_input"]
        + getattr(u, "output_tokens", 0) / 1_000_000 * IMAGE_PRICING["image_output"]
    )


# --- Overview ---
print("=" * 60)
print("GPT-IMAGE-2.5: Flare vs Sunburst")
print("=" * 60)
print()
print("Two sibling models, same token pricing, different quality/speed trade-offs:")
print("  gpt-image-2.5-flare    — Fast. Use for bulk generation, prototyping.")
print("  gpt-image-2.5-sunburst — Precise. Use for editing, text, diagrams.")
print()
print("New quality settings in 2.5 (not available in gpt-image-2):")
print("  standard | high | xhigh | max")
print()


# --- Example 1: Generation comparison (Flare vs Sunburst) ---
print("=" * 60)
print("EXAMPLE 1: Flare vs Sunburst — generation quality comparison")
print("=" * 60)
print()

PROMPT = (
    "A precise technical diagram showing three microservices (Auth, Orders, Inventory) "
    "connected by labeled arrows. Clean flat design. Blue, grey, and white only. "
    "Each service box has its name in 12pt sans-serif. Arrow labels: gRPC. "
    "White background, no shadows."
)

for model in ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]:
    import time
    t0 = time.time()
    response = client.images.generate(
        model=model,
        prompt=PROMPT,
        n=1,
        size="1024x1024",
        quality="high",
        response_format="b64_json",
    )
    elapsed = time.time() - t0

    filename = f"diagram_{model.split('-')[-1]}.png"
    image_bytes = base64.b64decode(response.data[0].b64_json)
    with open(filename, "wb") as f:
        f.write(image_bytes)

    c = image_cost(response)
    print(f"Model: {model}")
    print(f"  Saved:   {filename} ({len(image_bytes):,} bytes)")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Est. cost: ${c:.5f}")
    print()


# --- Example 2: xhigh quality (new in 2.5) ---
print("=" * 60)
print("EXAMPLE 2: xhigh quality setting (new in gpt-image-2.5)")
print("=" * 60)
print()

for quality in ["high", "xhigh"]:
    t0 = time.time()
    response = client.images.generate(
        model="gpt-image-2.5-sunburst",
        prompt=(
            "A photorealistic product shot of a glass water bottle on a white marble surface. "
            "Studio lighting. Label on bottle reads 'AQUA' in clean sans-serif. "
            "Soft shadows, high resolution."
        ),
        n=1,
        size="1024x1024",
        quality=quality,
        response_format="b64_json",
    )
    elapsed = time.time() - t0

    filename = f"product_{quality}.png"
    image_bytes = base64.b64decode(response.data[0].b64_json)
    with open(filename, "wb") as f:
        f.write(image_bytes)

    c = image_cost(response)
    print(f"Quality={quality}: latency={elapsed:.2f}s, size={len(image_bytes):,} bytes, cost=${c:.5f}")
    print(f"  Saved: {filename}")
print()


# --- Example 3: Editing with Sunburst ---
print("=" * 60)
print("EXAMPLE 3: Image editing with Sunburst (precision inpainting)")
print("=" * 60)
print()

# Use the diagram generated in Example 1 as base image
base_image_path = "diagram_sunburst.png"
if os.path.exists(base_image_path):
    with open(base_image_path, "rb") as f:
        base_image = f.read()

    # Edit: add a label to the arrows
    edit_response = client.images.edit(
        model="gpt-image-2.5-sunburst",
        image=base_image,
        prompt=(
            "Add a red 'DEPRECATED' watermark in the top-right corner. "
            "Keep the rest of the diagram exactly as-is."
        ),
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    edited_bytes = base64.b64decode(edit_response.data[0].b64_json)
    with open("diagram_edited.png", "wb") as f:
        f.write(edited_bytes)

    print(f"Edited image saved: diagram_edited.png ({len(edited_bytes):,} bytes)")
else:
    print(f"Skipping edit: {base_image_path} not found (run Example 1 first)")
print()


# --- Summary ---
print("=" * 60)
print("GPT-IMAGE-2.5 QUICK REFERENCE")
print("=" * 60)
print("""
Released:  September 8, 2026

Model IDs:
  gpt-image-2.5-flare     — Speed-optimized: everyday generation, high-volume
  gpt-image-2.5-sunburst  — Precision-optimized: editing, text rendering, diagrams

API endpoints:
  client.images.generate(model=..., prompt=..., quality=..., size=..., n=..., response_format=...)
  client.images.edit(model=..., image=..., prompt=..., ...)

Quality settings (new xhigh/max in 2.5 vs gpt-image-2 which capped at high):
  standard | high | xhigh | max
  xhigh/max increase generation time but produce notably sharper text and detail.

Pricing (Sep 2026) — token-based, same for both Flare and Sunburst:
  Image input:        $8.00 / 1M tokens
  Cached image input: $2.00 / 1M tokens
  Image output:      $30.00 / 1M tokens
  Text input:         $5.00 / 1M tokens
  Cached text input:  $1.25 / 1M tokens
  Batch API:          50% off (async workloads)

When to choose Flare vs Sunburst:
  Flare:    Creative generation, marketing assets, prototyping, high throughput
  Sunburst: Technical diagrams, infographics with text, inpainting, precision edits

Upgrade from gpt-image-2:
  - Same Images API endpoints; drop-in model ID swap
  - Add quality="xhigh" or quality="max" to unlock 2.5-only quality tier
  - Flare gives faster iteration; Sunburst gives gpt-image-2 users the best editing parity
""")

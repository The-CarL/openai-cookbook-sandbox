"""Exercise 40: GPT-Image-2.6 Flare and Sunburst — November 2026 image models.

Released November 4, 2026 (Images API: /v1/images/generations and /v1/images/edits):

  gpt-image-2.6-flare    — Speed tier.
  gpt-image-2.6-sunburst — Precision tier.

New in 2.6 (per the OpenAI image generation guide):
  - `complexity` ("low" | "medium" | "high", default "high"): how much the model
    plans before rendering. Separate from `quality`; no separate fee, but higher
    levels can add latency.
  - `output_format="svg"` (generations only, n=1, no streaming/partials,
    no output_compression). `quality` controls the internal reference image.
  - `output_format="gif"` (generations + edits) with `frame_count` (2-16, default 9)
    and `frame_duration_ms` (20-2000, default 200).
  - Markup-guided edits: send a transparent PNG annotation layer as `markup`
    alongside the original image (first image). Same size as the first image.
    Do not combine with `mask`.

Pricing (sunburst model page, Nov 2026): text in $5 / cached $1.25,
image in $8 / cached $2, image out $30 per 1M tokens (same as 2.5).
"""

import base64
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = "gpt-image-2.6-flare"


def save_b64(response, filename: str) -> None:
    data = base64.b64decode(response.data[0].b64_json)
    with open(filename, "wb") as f:
        f.write(data)
    print(f"  Saved {filename} ({len(data):,} bytes)")


# --- Example 1: complexity ---
print("=" * 60)
print("EXAMPLE 1: complexity low vs high")
print("=" * 60)
for complexity in ["low", "high"]:
    response = client.images.generate(
        model=MODEL,
        prompt="A clean infographic of the water cycle with labeled stages.",
        size="1024x1024",
        quality="medium",
        extra_body={"complexity": complexity},
    )
    print(f"complexity={complexity}")
    save_b64(response, f"water_cycle_{complexity}.png")
    print(f"  usage: {getattr(response, 'usage', None)}")


# --- Example 2: SVG output ---
print("=" * 60)
print("EXAMPLE 2: SVG output (generations only, n=1)")
print("=" * 60)
response = client.images.generate(
    model="gpt-image-2.6-sunburst",
    prompt="A minimal flat logo of a paper airplane, two colors.",
    n=1,
    output_format="svg",
)
save_b64(response, "logo.svg")


# --- Example 3: animated GIF ---
print("=" * 60)
print("EXAMPLE 3: animated GIF")
print("=" * 60)
response = client.images.generate(
    model=MODEL,
    prompt="A bouncing red ball on a white background, side view.",
    output_format="gif",
    size="512x512",  # multiples of 16, applies per frame
    extra_body={"frame_count": 8, "frame_duration_ms": 100},
)
save_b64(response, "ball.gif")


# --- Example 4: markup-guided edit ---
print("=" * 60)
print("EXAMPLE 4: markup-guided edit")
print("=" * 60)
if len(sys.argv) == 3:
    photo, overlay = sys.argv[1], sys.argv[2]  # overlay: transparent PNG, same size as photo
    with open(photo, "rb") as image, open(overlay, "rb") as markup:
        result = client.images.edit(
            model=MODEL,
            image=image,
            markup=markup,
            quality="medium",
            prompt=(
                "Recolor the object inside the red circle to solid cobalt blue. "
                "Keep everything else unchanged and remove the red circle."
            ),
        )
    save_b64(result, "markup_edit.png")
else:
    print("Skipped. Usage: python 40_image_gen_2_6.py photo.png markup_overlay.png")

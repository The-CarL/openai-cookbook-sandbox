"""Exercise 40: GPT-Image-2.6 Flare and Sunburst — November 2026 image models.

Released November 4, 2026 (OpenAI API changelog). Same two-tier split as 2.5:

  gpt-image-2.6-flare    — speed tier
  gpt-image-2.6-sunburst — precision tier (default snapshot)

Endpoints: v1/images/generations and v1/images/edits.

New in 2.6:
  - `complexity` ("low" | "medium" | "high", default "high"): how much the model
    plans before rendering. Low suits focused edits; high suits dense text and
    complex layouts (slower, may introduce unwanted changes). No separate fee.
    Separate from `quality`, which controls rendering quality.
  - Markup-guided editing (also on 2.5 models): send a transparent PNG
    annotation layer as `markup` to show WHERE to edit; describe the change in
    `prompt`. Must match the first image's width/height and contain transparent
    pixels. Do not combine with `mask`.
  - SVG and animated GIF output. The changelog confirms both; the exact
    `output_format` values weren't verified, so this exercise doesn't use them.
    Check the Images API reference before relying on them.

Pricing (token-based, per model page): text in $5 / cached $1.25,
image in $8 / cached $2, image out $30 per 1M.
"""

import base64
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PRICING = {"text_input": 5.00, "image_output": 30.00}


def cost(resp) -> float:
    u = getattr(resp, "usage", None)
    if not u:
        return 0.0
    return (
        getattr(u, "input_tokens", 0) / 1e6 * PRICING["text_input"]
        + getattr(u, "output_tokens", 0) / 1e6 * PRICING["image_output"]
    )


# --- Example 1: complexity vs latency ---
print("EXAMPLE 1: complexity low vs high")
PROMPT = "A paper-cut football scene with a trophy and a CHAMPIONS banner"
for complexity in ["low", "high"]:
    t0 = time.time()
    resp = client.images.generate(
        model="gpt-image-2.6-flare",
        prompt=PROMPT,
        complexity=complexity,
        quality="medium",
        size="1024x1024",
    )
    name = f"champions_{complexity}.png"
    with open(name, "wb") as f:
        f.write(base64.b64decode(resp.data[0].b64_json))
    print(f"  complexity={complexity}: {time.time() - t0:.2f}s, ~${cost(resp):.4f}, saved {name}")
print()

# --- Example 2: markup-guided edit ---
print("EXAMPLE 2: markup-guided edit")
BASE, OVERLAY = "kitchen.png", "kitchen_markup.png"  # OVERLAY: transparent PNG, same size as BASE
if os.path.exists(BASE) and os.path.exists(OVERLAY):
    with open(BASE, "rb") as image, open(OVERLAY, "rb") as markup:
        resp = client.images.edit(
            model="gpt-image-2.6-sunburst",
            image=(BASE, image, "image/png"),
            markup=(OVERLAY, markup, "image/png"),
            quality="medium",
            prompt=(
                "Recolor the object inside the red circle to solid cobalt blue. "
                "Keep everything else unchanged and remove the red circle."
            ),
        )
    with open("kitchen_edited.png", "wb") as f:
        f.write(base64.b64decode(resp.data[0].b64_json))
    print(f"  saved kitchen_edited.png, ~${cost(resp):.4f}")
else:
    print(f"  Skipping: provide {BASE} and a same-size transparent {OVERLAY}")
print()

print("""QUICK REFERENCE
  Models:  gpt-image-2.6-flare (speed), gpt-image-2.6-sunburst (precision)
  Params:  complexity=low|medium|high (default high), quality=..., markup=<transparent PNG>
  Rules:   markup and mask are mutually exclusive; no extra fee for complexity
  Upgrade: model-ID swap from 2.5; dense text -> complexity high, small edits -> low
""")

"""Exercise 40: GPT-Image-2.6 Flare and Sunburst — November 2026 image models.

Released November 4, 2026 (OpenAI API changelog). Same Images API endpoints as 2.5:

  gpt-image-2.6-flare    — speed tier
  gpt-image-2.6-sunburst — precision tier

New in 2.6:
  complexity      "low" | "medium" | "high" (default "high"); separate from `quality`.
                  Higher complexity can add latency and unwanted changes — compare.
  SVG output      output_format="svg" on /v1/images/generations only (n=1).
                  `quality` controls the internal reference image used to make the SVG.
  Animated GIF    output_format="gif" on generations or edits.
                  frame_count 2-16 (default 9), frame_duration_ms 20-2000 (default 200,
                  rounded to 10 ms), size "auto" or WxH in multiples of 16 (per frame).
  Markup edits    `markup` = transparent PNG annotation layer on /v1/images/edits
                  (also supported by 2.5). Must match first image's size; do not combine
                  with `mask`. Limits: 16,777,216 px, 50 MB.
Streaming, partial images and output_compression are unsupported for SVG/GIF.
Pricing (model page): same token rates as 2.5 — see exercise 36.

Parameters not yet in the SDK's typed signature go through `extra_body`.
"""

import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL = "gpt-image-2.6-flare"


def save(response, path: str, text: bool = False) -> None:
    raw = base64.b64decode(response.data[0].b64_json)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"Saved {path} ({len(raw):,} bytes)")


# Example 1: complexity
for level in ["low", "high"]:
    r = client.images.generate(
        model=MODEL,
        prompt="A cozy reading nook with plants and warm lamplight",
        size="1024x1024",
        quality="medium",
        extra_body={"complexity": level},
    )
    save(r, f"nook_{level}.png")

# Example 2: SVG logo
r = client.images.generate(
    model=MODEL,
    prompt="A simple bakery logo with a golden loaf",
    n=1,
    quality="medium",
    extra_body={"output_format": "svg"},
)
save(r, "bakery_logo.svg")

# Example 3: animated GIF
r = client.images.generate(
    model=MODEL,
    prompt="A paper plane gliding across a blue sky, looping animation",
    size="512x512",
    extra_body={"output_format": "gif", "frame_count": 9, "frame_duration_ms": 200},
)
save(r, "plane.gif")

# Example 4: markup-guided edit (annotation layer must have transparent pixels)
with open("kitchen.webp", "rb") as img, open("markup.png", "rb") as mk:
    r = client.images.edit(
        model=MODEL,
        image=img,
        prompt="Replace the circled object with a ceramic vase of tulips",
        quality="medium",
        extra_body={"markup": mk},
    )
save(r, "kitchen_edited.png")

"""Exercise 40: GPT-Image-2.6 — SVG, animated GIF, complexity, markup-guided edits.

Released November 4, 2026: gpt-image-2.6-sunburst (precision) and
gpt-image-2.6-flare (speed), on /v1/images/generations and /v1/images/edits.

New in 2.6:
  - output_format="svg"  (generations only, n=1) -> vector XML in data[0].b64_json
  - output_format="gif"  (generations only) -> animated GIF; frame_count (2-16,
    default 9), frame_duration_ms (20-2000, default 200), size ("auto" or WxH, multiples of 16)
  - complexity: "low" | "medium" | "high" (default "high"); no separate fee
  - markup: transparent PNG annotation layer for /v1/images/edits (must match the
    first image's dimensions; cannot combine with mask or prompt_rewriter)
  - quality: low | medium | high | xhigh | max

SVG and GIF: no streaming, partial images, output_compression, or prompt_rewriter.
Pricing: image output $30/M tokens; markup edits same rates as other edits. SVG is
billed for the reference image (image model) plus the tokens that write the SVG
(authoring model), so cost varies. Verify rates on the pricing page.
"""

import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: SVG output ---
print("EXAMPLE 1: SVG output")
svg = client.images.generate(
    model="gpt-image-2.6-sunburst",
    prompt="A flat vector logo of a lighthouse on a cliff, two colors, no gradients.",
    n=1,  # required for SVG
    quality="medium",  # controls the reference image made before the SVG
    output_format="svg",
)
with open("lighthouse.svg", "wb") as f:
    f.write(base64.b64decode(svg.data[0].b64_json))
print("Saved lighthouse.svg (static vector, no scripts or embedded rasters)")

# --- Example 2: Animated GIF ---
print("\nEXAMPLE 2: Animated GIF")
gif = client.images.generate(
    model="gpt-image-2.6-flare",
    prompt="A looping animation of a paper plane circling a desk lamp, flat illustration.",
    output_format="gif",
    frame_count=8,
    frame_duration_ms=120,
    size="512x512",
    quality="high",
)
with open("plane.gif", "wb") as f:
    f.write(base64.b64decode(gif.data[0].b64_json))
print("Saved plane.gif")

# --- Example 3: Complexity control ---
print("\nEXAMPLE 3: complexity low vs high")
for complexity in ["low", "high"]:
    r = client.images.generate(
        model="gpt-image-2.6-flare",
        prompt="A poster with the headline 'OPEN HOUSE' and a dense schedule table.",
        complexity=complexity,
        quality="medium",
    )
    with open(f"poster_{complexity}.png", "wb") as f:
        f.write(base64.b64decode(r.data[0].b64_json))
    print(f"complexity={complexity}: saved poster_{complexity}.png")

# --- Example 4: Markup-guided edit ---
print("\nEXAMPLE 4: markup-guided edit")
# markup.png: transparent PNG, same width/height as poster_high.png, with your
# annotations (circles, arrows) drawn on it.
try:
    with open("poster_high.png", "rb") as img, open("markup.png", "rb") as markup:
        edit = client.images.edit(
            model="gpt-image-2.6-flare",
            image=img,
            markup=markup,
            prompt="Change the circled text to 'CLOSED TODAY'. Leave everything else alone.",
            quality="medium",
        )
    with open("poster_edited.png", "wb") as f:
        f.write(base64.b64decode(edit.data[0].b64_json))
    print("Saved poster_edited.png")
except FileNotFoundError:
    print("Skipping: create markup.png (transparent annotation layer) first")

"""Exercise 40: GPT-Image-2.6 Flare and Sunburst — November 2026 image models.

Released November 4, 2026 (API changelog). Same Flare (speed) / Sunburst (precision)
split as 2.5, served from the Images API only (/v1/images/generations and
/v1/images/edits — NOT Responses or Chat Completions).

New in 2.6:
  - `complexity` request field: "low" | "medium" | "high" (default "high"), no extra fee
  - SVG generation and animated GIF generation through the Images API
  - Markup-guided editing: send a transparent PNG annotation layer as `markup`
    (2.5 and 2.6 models); don't combine with `mask`

Pricing (model page, same as 2.5), per 1M tokens:
  text in $5.00 (cached $1.25) | image in $8.00 (cached $2.00) | image out $30.00

The docs do not list the parameter names/values for SVG or GIF output, so this
exercise does not guess them — see the guide:
https://developers.openai.com/api/docs/guides/image-generation#gpt-image-26
"""

import base64
import io

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

load_dotenv()
client = OpenAI()

PRICING = {"text_input": 5.00, "image_output": 30.00}


def cost(response) -> float:
    u = getattr(response, "usage", None)
    if not u:
        return 0.0
    return (
        getattr(u, "input_tokens", 0) / 1e6 * PRICING["text_input"]
        + getattr(u, "output_tokens", 0) / 1e6 * PRICING["image_output"]
    )


# --- Example 1: complexity levels -------------------------------------------
print("EXAMPLE 1: complexity low vs high (no separate fee)")
for complexity in ["low", "high"]:
    r = client.images.generate(
        model="gpt-image-2.6-flare",
        prompt="A product icon of a paper airplane, flat design, white background.",
        size="1024x1024",
        quality="medium",
        extra_body={"complexity": complexity},  # new field; use extra_body on older SDKs
    )
    data = base64.b64decode(r.data[0].b64_json)
    with open(f"airplane_{complexity}.png", "wb") as f:
        f.write(data)
    print(f"  complexity={complexity}: {len(data):,} bytes, ~${cost(r):.5f}")


# --- Example 2: markup-guided edit ------------------------------------------
print("\nEXAMPLE 2: markup-guided edit")
base = Image.open("airplane_high.png").convert("RGBA")
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))  # transparent, same size as image
w, h = base.size
ImageDraw.Draw(overlay).ellipse(
    (w * 0.3, h * 0.3, w * 0.7, h * 0.7), outline=(255, 0, 0, 255), width=8
)
buf = io.BytesIO()
overlay.save(buf, format="PNG")  # keep as a separate layer; don't flatten

with open("airplane_high.png", "rb") as f:
    image_bytes = f.read()

result = client.images.edit(
    model="gpt-image-2.6-flare",
    image=("airplane.png", image_bytes, "image/png"),
    markup=("markup.png", buf.getvalue(), "image/png"),
    quality="medium",
    prompt="Recolor the object inside the red circle to solid cobalt blue. "
           "Remove the red circle and keep everything else unchanged.",
)
with open("airplane_edited.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
print(f"  saved airplane_edited.png, ~${cost(result):.5f}")

print("""
Markup rules: applies to the FIRST input image only; image and annotation each
<= 16,777,216 px; annotation file <= 50 MB; no `mask` alongside `markup`.
""")

"""Exercise 40: GPT-Image-2.6 Flare and Sunburst — November 2026 image models.

Released November 4, 2026 (Images API: /v1/images/generations, /v1/images/edits):

  gpt-image-2.6-flare    — Speed tier.
  gpt-image-2.6-sunburst — Precision tier.

New vs 2.5 (Exercise 36):
  complexity   — "low" | "medium" | "high" (default "high"). Separate from `quality`;
                 no extra fee. Low suits focused edits (e.g. swap a background);
                 high suits dense text / complex layouts but may add latency and
                 unwanted changes.
  markup       — Markup-guided edits on /v1/images/edits. Pass a transparent PNG
                 annotation layer (same width/height as the first image; each file
                 <= 16,777,216 px, annotation <= 50 MB). Cannot combine with `mask`.
                 Also supported on gpt-image-2.5 models.
  output_format="svg" — generations only, n=1, no streaming/partials/compression.
                 Billed: image model for the internal reference image + authoring
                 model tokens for the SVG.
  output_format="gif" — generations and edits. frame_count 2-16 (default 9),
                 frame_duration_ms 20-2000 (default 200), size "auto" or WxH
                 (multiples of 16). No streaming/partials/output_compression.

Pricing (same token rates as 2.5): text in $5 (cached $1.25), image in $8
(cached $2), image out $30 per 1M tokens.

Note: the extra-body fields below follow the changelog / image-generation guide;
older SDK versions may not expose them as typed kwargs, so we pass them via
`extra_body`.
"""

import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = "gpt-image-2.6-sunburst"


def save(resp, path: str) -> None:
    data = base64.b64decode(resp.data[0].b64_json)
    with open(path, "wb") as f:
        f.write(data)
    print(f"  saved {path} ({len(data):,} bytes)")


# --- Example 1: complexity control ---
print("EXAMPLE 1: complexity low vs high")
for complexity in ["low", "high"]:
    resp = client.images.generate(
        model=MODEL,
        prompt="A product label with a short ingredient list and a barcode",
        size="1024x1024",
        quality="medium",
        extra_body={"complexity": complexity},
    )
    save(resp, f"label_{complexity}.png")

# --- Example 2: SVG output ---
print("\nEXAMPLE 2: SVG output (generations only, n=1)")
resp = client.images.generate(
    model=MODEL,
    prompt="A flat vector logo: a paper plane circling a globe, two colors",
    n=1,
    quality="medium",
    extra_body={"output_format": "svg"},
)
save(resp, "logo.svg")  # UTF-8 XML, static, no scripts/external resources

# --- Example 3: animated GIF ---
print("\nEXAMPLE 3: animated GIF")
resp = client.images.generate(
    model="gpt-image-2.6-flare",
    prompt="A looping animation of a bouncing ball with a soft shadow",
    extra_body={
        "output_format": "gif",
        "frame_count": 8,
        "frame_duration_ms": 120,
        "size": "auto",
    },
)
save(resp, "ball.gif")

# --- Example 4: markup-guided edit ---
print("\nEXAMPLE 4: markup-guided edit")
# `markup` is a transparent PNG annotation layer (same size as the first image).
# Don't combine with `mask`. Typed SDK support may lag; fall back to raw HTTP
# (multipart field named `markup`) if your openai version rejects it.
try:
    with open("photo.png", "rb") as photo, open("markup.png", "rb") as markup:
        resp = client.images.edit(
            model=MODEL,
            image=photo,
            prompt="Recolor the circled vase to deep blue",
            quality="medium",
            extra_body={"markup": markup},
        )
    save(resp, "edited.png")
except FileNotFoundError:
    print("  skipped: photo.png / markup.png not found")

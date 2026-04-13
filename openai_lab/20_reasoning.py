"""Exercise 20: GPT-5.4 reasoning effort — built-in chain-of-thought control."""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

PROMPT = (
    "A customer has a RAG pipeline with 3 retrieval stages: "
    "sparse keyword search (BM25), dense vector search (cosine similarity), "
    "and a reranker (cross-encoder). Their p99 latency went from 200ms to 800ms "
    "after adding the reranker. The reranker processes 50 candidates. "
    "They want to keep the reranker for quality but need p99 under 400ms. "
    "What's the optimal architecture change?"
)

# GPT-5.4 supports: none, low, medium, high
# (xhigh also exists but is very slow/expensive — omitted here)
EFFORTS = ["none", "low", "medium", "high"]

results = []

for effort in EFFORTS:
    print(f"\n{'='*60}")
    print(f"REASONING EFFORT: {effort}")
    print(f"{'='*60}")

    start = time.time()
    response = client.responses.create(
        model="gpt-5.4",
        input=PROMPT,
        reasoning={"effort": effort},
    )
    elapsed = time.time() - start

    results.append({
        "effort": effort,
        "elapsed": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "text": response.output_text,
    })

    # Show truncated response
    text_preview = response.output_text[:400]
    if len(response.output_text) > 400:
        text_preview += "..."
    print(f"\n{text_preview}")
    print(f"\n--- Stats ---")
    print(f"Latency: {elapsed:.2f}s")
    print(f"Tokens:  {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# --- Summary comparison ---
print("\n" + "=" * 60)
print("REASONING EFFORT COMPARISON")
print("=" * 60)
print(f"{'Effort':<10} {'Latency':>8} {'In tok':>8} {'Out tok':>8} {'Total':>8}")
print("-" * 50)
for r in results:
    print(f"{r['effort']:<10} {r['elapsed']:>7.2f}s {r['input_tokens']:>8} {r['output_tokens']:>8} {r['total_tokens']:>8}")

# --- Compare with dedicated reasoning model ---
print("\n" + "=" * 60)
print("BONUS: GPT-5.4 (high) vs o4-mini (dedicated reasoning model)")
print("=" * 60)

start = time.time()
o4_response = client.responses.create(
    model="o4-mini",
    input=PROMPT,
    reasoning={"effort": "medium"},
)
o4_elapsed = time.time() - start

print(f"\no4-mini response ({o4_elapsed:.2f}s):")
text_preview = o4_response.output_text[:400]
if len(o4_response.output_text) > 400:
    text_preview += "..."
print(text_preview)
print(f"\nTokens: {o4_response.usage.input_tokens} in, {o4_response.usage.output_tokens} out")

gpt54_high = results[3]  # "high" effort result
print(f"\n--- Comparison ---")
print(f"{'Model':<25} {'Latency':>8} {'Output tok':>10}")
print("-" * 48)
print(f"{'gpt-5.4 (effort=high)':<25} {gpt54_high['elapsed']:>7.2f}s {gpt54_high['output_tokens']:>10}")
print(f"{'o4-mini (effort=medium)':<25} {o4_elapsed:>7.2f}s {o4_response.usage.output_tokens:>10}")

print("\n--- When to use each ---")
print("GPT-5.4 reasoning:  General tasks that sometimes need deeper thinking.")
print("                    Adjust effort per-request. One model for everything.")
print("o4-mini:            Dedicated reasoning model. Math, logic, code analysis.")
print("                    Better for consistently hard problems. $1.10/$4.40 per 1M.")
print("o3:                 Most powerful reasoning. Complex proofs, research.")
print("                    Same price as gpt-4.1 ($2.00/$8.00) but slower.")

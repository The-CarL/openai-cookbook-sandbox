"""Exercise 14: Generate embeddings — compare models and dimension reduction."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "OpenAI's Responses API is the recommended primitive for building AI applications."

# --- text-embedding-3-small (default 1536 dims) ---
print("=== text-embedding-3-small ===")
r_small = client.embeddings.create(model="text-embedding-3-small", input=text)
emb_small = r_small.data[0].embedding
print(f"Default dimensions: {len(emb_small)}")
print(f"First 5 values: {emb_small[:5]}")
print(f"Usage: {r_small.usage.total_tokens} tokens")

# Dimension reduction — same model, fewer dimensions
r_small_256 = client.embeddings.create(model="text-embedding-3-small", input=text, dimensions=256)
emb_small_256 = r_small_256.data[0].embedding
print(f"\nWith dimensions=256: {len(emb_small_256)} dims")
print(f"First 5 values: {emb_small_256[:5]}")

# --- text-embedding-3-large (default 3072 dims) ---
print("\n=== text-embedding-3-large ===")
r_large = client.embeddings.create(model="text-embedding-3-large", input=text)
emb_large = r_large.data[0].embedding
print(f"Default dimensions: {len(emb_large)}")
print(f"First 5 values: {emb_large[:5]}")
print(f"Usage: {r_large.usage.total_tokens} tokens")

# Dimension reduction to 1024
r_large_1024 = client.embeddings.create(model="text-embedding-3-large", input=text, dimensions=1024)
emb_large_1024 = r_large_1024.data[0].embedding
print(f"\nWith dimensions=1024: {len(emb_large_1024)} dims")

# Dimension reduction to 256
r_large_256 = client.embeddings.create(model="text-embedding-3-large", input=text, dimensions=256)
emb_large_256 = r_large_256.data[0].embedding
print(f"With dimensions=256: {len(emb_large_256)} dims")

print("\n=== Comparison ===")
print(f"{'Model':<28} {'Dims':>6} {'Cost/1M tokens':>15}")
print("-" * 55)
print(f"{'text-embedding-3-small':<28} {'1536':>6} {'$0.02':>15}")
print(f"{'text-embedding-3-small':<28} {'256':>6} {'$0.02':>15} (same cost, less storage)")
print(f"{'text-embedding-3-large':<28} {'3072':>6} {'$0.13':>15}")
print(f"{'text-embedding-3-large':<28} {'1024':>6} {'$0.13':>15} (same cost, less storage)")
print(f"{'text-embedding-3-large':<28} {'256':>6} {'$0.13':>15} (same cost, less storage)")
print()
print("Key insight: Dimension reduction is FREE — you pay the same token cost.")
print("The tradeoff is retrieval quality vs storage/compute. For most use cases,")
print("text-embedding-3-small at 1536 dims is the cost-effective default.")
print("Reduce to 256 dims if you need to save on vector DB storage costs.")

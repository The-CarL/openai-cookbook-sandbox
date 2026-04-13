"""Exercise 15: Similarity search — the foundation of what file_search does."""

import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


# --- Build a knowledge base ---
documents = [
    # Technical topics
    "The Responses API supports built-in tools like web_search, file_search, and code_interpreter.",
    "Function calling allows LLMs to interact with external systems through structured JSON schemas.",
    "Structured outputs with strict mode guarantee the response matches your JSON schema exactly.",
    "Vector stores enable semantic search over uploaded documents for RAG applications.",
    "The previous_response_id parameter maintains conversation context without manual message management.",
    # Enterprise topics
    "OpenAI offers SOC 2 Type II certification and HIPAA BAA for enterprise customers.",
    "Data residency options are available in US, EU, UK, Japan, Australia, and other regions.",
    "SCIM provisioning enables automated user lifecycle management with identity providers.",
    "Enterprise customers get dedicated support, custom rate limits, and usage analytics.",
    # Pricing/business topics
    "GPT-4.1-mini costs $0.40 per million input tokens and $1.60 per million output tokens.",
    "Text-embedding-3-small costs $0.02 per million tokens with 1536 default dimensions.",
    "The dimension reduction feature lets you trade retrieval quality for storage savings at no extra cost.",
    # Model selection
    "GPT-4.1-nano is the fastest and cheapest model, ideal for classification and routing.",
    "GPT-4.1 offers the best quality for complex reasoning and nuanced generation tasks.",
    "Use o4-mini, o3, or GPT-5.4 with reasoning effort for tasks requiring extended reasoning chains.",
]

print("=== Embedding documents ===")
doc_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents,
)
doc_embeddings = [d.embedding for d in doc_response.data]
print(f"Embedded {len(documents)} documents ({doc_response.usage.total_tokens} tokens)")

# --- Query ---
queries = [
    "How do I ensure my API responses match a specific schema?",
    "What compliance certifications does OpenAI have?",
    "Which model should I use if I need the cheapest option?",
    "How does conversation memory work in the Responses API?",
]

print("\n" + "=" * 70)
print("SIMILARITY SEARCH RESULTS")
print("=" * 70)

for query in queries:
    q_response = client.embeddings.create(model="text-embedding-3-small", input=query)
    q_embedding = q_response.data[0].embedding

    # Compute similarity to all documents
    scores = [(i, cosine_similarity(q_embedding, doc_emb)) for i, doc_emb in enumerate(doc_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nQuery: \"{query}\"")
    print(f"Top 3 results:")
    for rank, (idx, score) in enumerate(scores[:3], 1):
        print(f"  {rank}. [{score:.4f}] {documents[idx]}")

    # Show the gap between top result and worst result
    print(f"  Score range: {scores[0][1]:.4f} (best) to {scores[-1][1]:.4f} (worst)")

print("\n=== What this teaches us ===")
print("This is exactly what file_search does under the hood:")
print("1. Documents are chunked and embedded")
print("2. Queries are embedded with the same model")
print("3. Cosine similarity ranks the most relevant chunks")
print("4. Top chunks are injected into the LLM's context for generation")
print("The difference: file_search also handles chunking, vector storage, and retrieval optimization.")

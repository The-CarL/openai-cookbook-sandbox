"""Exercise 7: File search with vector stores — enterprise RAG pattern."""

import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Step 1: Create realistic enterprise documents ---
docs = {
    "product_overview.txt": """CloudSync Enterprise Platform - Product Overview

CloudSync is a cloud-native data synchronization platform designed for Fortune 500 companies.
Key features include:
- Real-time bidirectional data sync across 40+ enterprise systems
- Custom transformation rules with a no-code builder
- Enterprise-grade encryption (AES-256, TLS 1.3)
- SOC 2 Type II and HIPAA compliant
- 99.99% uptime SLA

Pricing tiers:
- Starter: $5,000/month (up to 1M records/day)
- Professional: $15,000/month (up to 10M records/day)
- Enterprise: Custom pricing (unlimited records, dedicated support)

Supported integrations: Salesforce, SAP, Oracle, Workday, ServiceNow, Snowflake, BigQuery, Databricks.
""",
    "troubleshooting_guide.txt": """CloudSync Troubleshooting Guide

Common Issues and Resolutions:

1. SYNC LATENCY ABOVE SLA
   - Check the sync queue depth in the admin dashboard
   - Verify source system API rate limits haven't changed
   - Review transformation rule complexity (rules with >5 lookups can cause 3x latency)
   - Escalation: If p99 latency exceeds 500ms for >1 hour, page the on-call engineer

2. DATA MISMATCH ERRORS
   - Enable field-level audit logging
   - Check schema drift between source and destination
   - Verify custom transformation rules haven't been modified recently
   - Common cause: Timezone handling in date fields (always use UTC internally)

3. AUTHENTICATION FAILURES
   - OAuth tokens expire every 60 minutes; check token refresh logic
   - For SAML-based SSO, verify IdP metadata hasn't changed
   - Service accounts should use API keys, not user credentials
   - Rate limit for auth endpoints: 100 requests/minute per tenant

4. CONNECTOR TIMEOUTS
   - Default timeout: 30 seconds. Can be increased to 120s for SAP and Oracle
   - Check network connectivity between CloudSync and customer VPC
   - For Snowflake: ensure warehouse is not suspended (auto-resume adds 10-30s)
""",
    "security_compliance.txt": """CloudSync Security & Compliance Documentation

Data Handling:
- All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- Zero data retention: CloudSync processes data in memory, never persists customer data
- Data residency: Available in US, EU, UK, Japan, and Australia regions
- Customer-managed encryption keys (CMEK) available on Enterprise tier

Compliance Certifications:
- SOC 2 Type II (audited annually by Deloitte)
- HIPAA BAA available for healthcare customers
- GDPR compliant with DPA available
- ISO 27001 certified
- FedRAMP Moderate (in progress, expected Q3 2026)

Access Controls:
- RBAC with 5 predefined roles (Admin, Developer, Analyst, Viewer, Auditor)
- SSO via SAML 2.0 and OIDC
- SCIM provisioning for automated user lifecycle management
- Audit logs retained for 7 years (configurable)
- IP allowlisting available on Professional and Enterprise tiers

Incident Response:
- 24/7 security operations center
- Mean time to detect (MTTD): <15 minutes
- Mean time to respond (MTTR): <1 hour for critical incidents
- Customer notification within 72 hours per GDPR requirements
""",
}

# --- Step 2: Create a vector store ---
print("=== Creating vector store ===")
vector_store = client.vector_stores.create(name="CloudSync Enterprise Docs")
print(f"Vector store ID: {vector_store.id}")
print(f"Status: {vector_store.status}")

# --- Step 3: Upload files and add to vector store ---
print("\n=== Uploading files ===")
file_ids = []
for filename, content in docs.items():
    # Upload the file
    file_obj = client.files.create(
        file=(filename, content.encode()),
        purpose="assistants",  # Required purpose for vector store files
    )
    file_ids.append(file_obj.id)
    print(f"Uploaded {filename} -> {file_obj.id}")

# Add files to vector store
print("\n=== Adding files to vector store ===")
for fid in file_ids:
    client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=fid)
    print(f"Added {fid} to vector store")

# Wait for processing
print("\n=== Waiting for vector store processing ===")
while True:
    vs = client.vector_stores.retrieve(vector_store.id)
    counts = vs.file_counts
    print(f"Status: {vs.status} | completed={counts.completed}, in_progress={counts.in_progress}, failed={counts.failed}")
    if counts.in_progress == 0:
        break
    time.sleep(1)

print(f"\nVector store ready! {counts.completed} files indexed.")

# --- Step 4: Query the vector store via file_search ---
print("\n" + "=" * 60)
print("QUERY 1: Technical troubleshooting")
print("=" * 60)
r1 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
    input="A customer is experiencing sync latency above their SLA. What should they check first, and when should they escalate?",
)
print(f"\n{r1.output_text}")

# Show citations
for item in r1.output:
    if item.type == "message":
        for content in item.content:
            if hasattr(content, "annotations") and content.annotations:
                print(f"\n--- Citations ({len(content.annotations)}) ---")
                for ann in content.annotations[:5]:
                    if hasattr(ann, "filename"):
                        print(f"  Source: {ann.filename}")
                    if hasattr(ann, "url"):
                        print(f"  URL: {ann.url}")

print(f"\nTokens: {r1.usage.input_tokens} in, {r1.usage.output_tokens} out")

print("\n" + "=" * 60)
print("QUERY 2: Security/compliance question")
print("=" * 60)
r2 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
    input="Does CloudSync support HIPAA? What about FedRAMP? Can we get a BAA?",
)
print(f"\n{r2.output_text}")
print(f"\nTokens: {r2.usage.input_tokens} in, {r2.usage.output_tokens} out")

print("\n" + "=" * 60)
print("QUERY 3: Pricing question (tests retrieval accuracy)")
print("=" * 60)
r3 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
    input="What are the pricing tiers and what's included in each?",
)
print(f"\n{r3.output_text}")
print(f"\nTokens: {r3.usage.input_tokens} in, {r3.usage.output_tokens} out")

# --- Step 5: Cleanup ---
print("\n=== Cleaning up ===")
# Delete vector store (this also removes file associations)
client.vector_stores.delete(vector_store.id)
print(f"Deleted vector store {vector_store.id}")

# Delete uploaded files
for fid in file_ids:
    client.files.delete(fid)
    print(f"Deleted file {fid}")

print("\nDone! All resources cleaned up.")

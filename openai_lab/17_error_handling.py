"""Exercise 17: Error handling — common failure modes and resilience patterns."""

import json

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, BadRequestError, AuthenticationError

load_dotenv()

client = OpenAI()


def test_error(name, func):
    """Run a function and display the error details."""
    print(f"\n--- {name} ---")
    try:
        func()
        print("  (No error — unexpected!)")
    except RateLimitError as e:
        print(f"  Error type: RateLimitError (HTTP {e.status_code})")
        print(f"  Message: {e.message[:120]}")
        print(f"  Retry strategy: Exponential backoff, check Retry-After header")
    except BadRequestError as e:
        print(f"  Error type: BadRequestError (HTTP {e.status_code})")
        print(f"  Message: {e.message[:200]}")
        print(f"  Fix: Check your request parameters")
    except AuthenticationError as e:
        print(f"  Error type: AuthenticationError (HTTP {e.status_code})")
        print(f"  Message: {e.message[:120]}")
        print(f"  Fix: Check your API key")
    except APIError as e:
        print(f"  Error type: {type(e).__name__} (HTTP {e.status_code})")
        print(f"  Message: {e.message[:200]}")
    except Exception as e:
        print(f"  Error type: {type(e).__name__}")
        print(f"  Message: {str(e)[:200]}")


print("=" * 60)
print("COMMON ERROR MODES")
print("=" * 60)

# 1. Invalid model name
test_error("Invalid model name", lambda: client.responses.create(
    model="gpt-5-turbo-nonexistent",
    input="Hello",
))

# 2. Malformed function schema (missing required field)
test_error("Malformed function schema", lambda: client.responses.create(
    model="gpt-4.1-mini",
    input="Hello",
    tools=[{
        "type": "function",
        "name": "bad_func",
        # Missing "parameters" and "strict"
    }],
))

# 3. Invalid structured output schema
test_error("Invalid structured output schema", lambda: client.responses.create(
    model="gpt-4.1-mini",
    input="Hello",
    text={"format": {
        "type": "json_schema",
        "name": "test",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                # With strict, additionalProperties: False is required
            },
        },
    }},
))

# 4. Context window overflow (simulate with huge input)
test_error("Very long input (context test)", lambda: client.responses.create(
    model="gpt-4.1-mini",
    input="x " * 500_000,  # Way over context limit
    max_output_tokens=10,
))

# 5. Empty input
test_error("Empty input", lambda: client.responses.create(
    model="gpt-4.1-mini",
    input="",
))

# --- Resilience pattern ---
print("\n" + "=" * 60)
print("RESILIENCE PATTERN: Retry with backoff")
print("=" * 60)
print("""
from openai import OpenAI, RateLimitError
import time

client = OpenAI()

def call_with_retry(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
            print(f"Rate limited. Retrying in {delay}s...")
            time.sleep(delay)

# Usage:
# result = call_with_retry(
#     lambda: client.responses.create(model="gpt-4.1-mini", input="Hello")
# )
""")

print("=" * 60)
print("ERROR HANDLING CHEAT SHEET")
print("=" * 60)
print("""
Error                  | HTTP | Cause                    | Action
-----------------------|------|--------------------------|---------------------------
AuthenticationError    | 401  | Bad API key              | Check OPENAI_API_KEY
RateLimitError         | 429  | Too many requests        | Exponential backoff
BadRequestError        | 400  | Invalid params/schema    | Fix the request
NotFoundError          | 404  | Invalid model/resource   | Check model name
APIError               | 500  | Server-side issue        | Retry after delay
APIConnectionError     | -    | Network issue            | Check connectivity, retry

Key tips for enterprise:
- Always use try/except around API calls
- Implement exponential backoff for 429s
- Log error types and messages for debugging
- Set max_output_tokens to prevent runaway costs
- Use strict: true schemas to catch errors at compile time, not runtime
""")

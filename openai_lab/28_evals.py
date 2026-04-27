"""Exercise 28: Evals — measuring whether a prompt actually works.

Every Solutions Engineering conversation eventually lands on the same
question: "how do I know if this prompt is good enough?" Evals are the
answer. This exercise shows two patterns:

  1. Programmatic eval — exact-match / regex / structured-output checks.
     Fast, cheap, no judge model needed. Use for anything verifiable.
  2. LLM-as-judge eval — score outputs on rubrics that aren't checkable
     by code (helpfulness, tone, faithfulness to context). Use sparingly
     because it's slow and expensive.

Run both over a fixed dataset, aggregate, and you have a regression test
suite for prompts. This is how you compare gpt-5.4-mini vs gpt-5.5 with
real numbers instead of vibes.
"""

import json
import statistics

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()


# === The eval dataset ===
# Each case has the input the prompt will see and a `truth` block describing
# what a correct answer must contain. Keep this small and high-signal.

DATASET = [
    {
        "id": "ext-1",
        "input": "Schedule a call with priya@globex.com for Tuesday April 28 2026 at 2pm PT to review the Q2 renewal.",
        "truth": {
            "must_contain_email": "priya@globex.com",
            "must_contain_date": "2026-04-28",
            "intent": "schedule_meeting",
        },
    },
    {
        "id": "ext-2",
        "input": "Cancel the integration kickoff with marcus.r@initech.io originally set for May 3.",
        "truth": {
            "must_contain_email": "marcus.r@initech.io",
            "must_contain_date": "2026-05-03",
            "intent": "cancel_meeting",
        },
    },
    {
        "id": "ext-3",
        "input": "Forward the SOC 2 report to security@acme.com — they need it by EOD Friday May 1.",
        "truth": {
            "must_contain_email": "security@acme.com",
            "must_contain_date": "2026-05-01",
            "intent": "send_document",
        },
    },
    {
        "id": "ext-4",
        "input": "Loop wayne-eng@waynetech.com into the perf review thread tomorrow morning.",
        "truth": {
            "must_contain_email": "wayne-eng@waynetech.com",
            # No specific date provided — model should infer relative to context
            "intent": "share_thread",
        },
    },
]


class Extraction(BaseModel):
    intent: str
    email: str
    date: str | None
    summary: str


# Structured output schema (strict mode)
SCHEMA = {
    "type": "json_schema",
    "name": "extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["schedule_meeting", "cancel_meeting", "send_document", "share_thread", "other"],
            },
            "email": {"type": "string"},
            "date": {"type": ["string", "null"], "description": "ISO date or null"},
            "summary": {"type": "string"},
        },
        "required": ["intent", "email", "date", "summary"],
        "additionalProperties": False,
    },
}


def run_extraction(model: str, text: str) -> Extraction:
    response = client.responses.create(
        model=model,
        instructions=(
            "Extract the structured action from a CSM message. "
            "Return ISO date if present, else null."
        ),
        input=text,
        text={"format": SCHEMA},
    )
    return Extraction(**json.loads(response.output_text))


# === Pattern 1: Programmatic eval ===

def programmatic_score(case, output: Extraction) -> dict:
    """Exact-match checks against the truth block."""
    checks = {}
    truth = case["truth"]
    if "must_contain_email" in truth:
        checks["email_correct"] = output.email == truth["must_contain_email"]
    if "must_contain_date" in truth:
        checks["date_correct"] = output.date == truth["must_contain_date"]
    if "intent" in truth:
        checks["intent_correct"] = output.intent == truth["intent"]
    checks["passed"] = all(checks.values())
    return checks


# === Pattern 2: LLM-as-judge eval ===

JUDGE_SCHEMA = {
    "type": "json_schema",
    "name": "judge",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "description": "1-5"},
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
        "additionalProperties": False,
    },
}


def llm_judge(input_text: str, output: Extraction) -> dict:
    """Rate the summary's helpfulness on a 1-5 rubric."""
    judge_response = client.responses.create(
        model="gpt-5.4-mini",  # Cheaper judge — perfectly fine for this rubric
        instructions=(
            "Rate the summary 1-5 on whether it's a clear, faithful one-line "
            "restatement of the input. 5 = perfect; 1 = misleading or missing key facts."
        ),
        input=f"INPUT:\n{input_text}\n\nSUMMARY:\n{output.summary}",
        text={"format": JUDGE_SCHEMA},
    )
    return json.loads(judge_response.output_text)


# === Run the eval across two models ===

MODELS_UNDER_TEST = ["gpt-5.4-mini", "gpt-5.5"]

print("=" * 70)
print("EVAL RUN — comparing models on extraction task")
print("=" * 70)

results = {model: [] for model in MODELS_UNDER_TEST}

for model in MODELS_UNDER_TEST:
    print(f"\n--- {model} ---")
    for case in DATASET:
        try:
            output = run_extraction(model, case["input"])
            programmatic = programmatic_score(case, output)
            judge = llm_judge(case["input"], output)
            row = {
                "case_id": case["id"],
                "passed": programmatic["passed"],
                "judge_score": judge["score"],
                "extracted_email": output.email,
                "extracted_intent": output.intent,
            }
        except Exception as e:
            row = {"case_id": case["id"], "passed": False, "judge_score": 0, "error": str(e)[:80]}
        results[model].append(row)
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"  [{flag}] {case['id']:<8} judge={row['judge_score']}  "
              f"intent={row.get('extracted_intent', '?')}")

# === Aggregate ===

print("\n" + "=" * 70)
print("EVAL SUMMARY")
print("=" * 70)
print(f"\n{'Model':<18} {'Pass rate':>10} {'Avg judge':>10}")
print("-" * 42)
for model, rows in results.items():
    pass_rate = sum(r["passed"] for r in rows) / len(rows)
    judge_scores = [r["judge_score"] for r in rows if r["judge_score"] > 0]
    avg_judge = statistics.mean(judge_scores) if judge_scores else 0
    print(f"{model:<18} {pass_rate:>9.0%} {avg_judge:>10.2f}")

# === Eval design notes ===
print("\n" + "=" * 70)
print("EVAL DESIGN NOTES")
print("=" * 70)
print("""
The hardest part of evals is the dataset, not the harness.

1. Start with 10-30 cases. Resist the urge to scale before the rubric is right.
   Bad data + 10K cases = false confidence. Good data + 20 cases = signal.

2. Lock truth carefully. If two reasonable humans disagree on the answer,
   the case is a poor eval target — drop it or split it.

3. Prefer programmatic checks. LLM-as-judge is necessary for fluency / tone
   but is itself a model that drifts. Pin the judge model + judge prompt and
   re-run baselines whenever you change either.

4. Evaluate in CI. Block prompt PRs that regress pass rate by > N%. This is
   the single best habit for a prompt-heavy codebase.

5. OpenAI has a hosted Evals product (client.evals.* in the SDK) for running
   these at scale with versioned datasets and dashboards. Use it once your
   eval suite outgrows a single script.

6. Things to eval beyond accuracy:
   - Cost per case (mean tokens × price)
   - Latency p50/p95
   - Refusal rate / safety policy compliance
   - Tool-call validity (right function, right args, right order)
""")

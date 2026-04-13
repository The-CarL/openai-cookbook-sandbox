"""Exercise 10: Complex nested schema — enterprise onboarding checklist."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

schema = {
    "type": "json_schema",
    "name": "onboarding_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string"},
            "plan_created": {"type": "string"},
            "estimated_go_live": {"type": "string"},
            "overall_status": {
                "type": "string",
                "enum": ["on_track", "at_risk", "delayed", "completed"],
            },
            "phases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phase_name": {"type": "string"},
                        "phase_number": {"type": "integer"},
                        "status": {
                            "type": "string",
                            "enum": ["not_started", "in_progress", "completed", "blocked"],
                        },
                        "target_completion": {"type": "string"},
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_name": {"type": "string"},
                                    "owner": {"type": "string"},
                                    "owner_type": {
                                        "type": "string",
                                        "enum": ["customer", "vendor", "shared"],
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["todo", "in_progress", "done", "blocked"],
                                    },
                                    "blockers": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "notes": {"type": "string"},
                                },
                                "required": [
                                    "task_name",
                                    "owner",
                                    "owner_type",
                                    "status",
                                    "blockers",
                                    "notes",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "phase_name",
                        "phase_number",
                        "status",
                        "target_completion",
                        "tasks",
                    ],
                    "additionalProperties": False,
                },
            },
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "mitigation": {"type": "string"},
                    },
                    "required": ["risk", "severity", "mitigation"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "customer_name",
            "plan_created",
            "estimated_go_live",
            "overall_status",
            "phases",
            "risks",
        ],
        "additionalProperties": False,
    },
}

response = client.responses.create(
    model="gpt-4.1-mini",
    input="""Generate a realistic enterprise onboarding plan for:

Customer: Pacific Northwest Healthcare (300-bed hospital network)
Product: CloudSync Enterprise tier
Context: They need HIPAA-compliant data sync between their Epic EHR, Workday HR, and Snowflake analytics warehouse.
Special requirements: BAA must be signed before any PHI touches the system. Their IT team is small (3 people) so they need heavy vendor support.
Timeline: They want to be live in 8 weeks.

Create a 4-phase onboarding plan with realistic tasks, owners, and potential blockers.
""",
    text={"format": schema},
)

result = json.loads(response.output_text)

# Pretty display
print("=== Enterprise Onboarding Plan ===\n")
print(f"Customer: {result['customer_name']}")
print(f"Created:  {result['plan_created']}")
print(f"Go-live:  {result['estimated_go_live']}")
print(f"Status:   {result['overall_status']}")

for phase in result["phases"]:
    print(f"\n--- Phase {phase['phase_number']}: {phase['phase_name']} ({phase['status']}) ---")
    print(f"    Target: {phase['target_completion']}")
    for task in phase["tasks"]:
        blocker_flag = " [BLOCKED]" if task["blockers"] else ""
        print(f"    [{task['status']:>11}] {task['task_name']} (owner: {task['owner']}, {task['owner_type']}){blocker_flag}")
        if task["blockers"]:
            for b in task["blockers"]:
                print(f"               Blocker: {b}")

print(f"\n--- Risks ---")
for risk in result["risks"]:
    print(f"  [{risk['severity']:>6}] {risk['risk']}")
    print(f"          Mitigation: {risk['mitigation']}")

# Validate schema compliance
print(f"\n=== Schema Compliance ===")
total_tasks = sum(len(p["tasks"]) for p in result["phases"])
print(f"Phases: {len(result['phases'])}")
print(f"Total tasks: {total_tasks}")
print(f"Risks: {len(result['risks'])}")

# Check all enums
for phase in result["phases"]:
    assert phase["status"] in ["not_started", "in_progress", "completed", "blocked"]
    for task in phase["tasks"]:
        assert task["status"] in ["todo", "in_progress", "done", "blocked"]
        assert task["owner_type"] in ["customer", "vendor", "shared"]
for risk in result["risks"]:
    assert risk["severity"] in ["low", "medium", "high"]
print("All enums valid!")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

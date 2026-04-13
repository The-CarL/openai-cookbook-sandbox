"""Exercise 6: Code interpreter tool — data analysis in a sandbox."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    tools=[{
        "type": "code_interpreter",
        "container": {"type": "auto"},
    }],
    instructions=(
        "Use the code_interpreter tool for all calculations. "
        "Use print() to display results clearly. "
        "After the code runs, present a concise final summary."
    ),
    input="""Analyze this CSV data using code_interpreter:

Customer,Tickets_Q1,Tickets_Q2,Tickets_Q3,Tickets_Q4,Contract_Value,Churn_Risk
Acme Corp,12,8,15,22,500000,High
Globex Inc,3,2,4,3,1200000,Low
Initech,7,12,18,25,300000,Critical
Wayne Enterprises,1,1,2,1,2000000,Low
Umbrella Corp,5,8,6,9,800000,Medium
Stark Industries,2,3,2,4,1500000,Low
Cyberdyne,20,25,30,35,400000,Critical

Calculate: total tickets, trend (increasing/decreasing/stable), tickets per $100K contract value.
Identify the top 3 accounts needing immediate attention. Print a summary table.
""",
)

# Show the full output flow
print("=== Output flow ===\n")
all_text_parts = []
for i, item in enumerate(response.output):
    if item.type == "message":
        for content in item.content:
            if content.type == "output_text":
                all_text_parts.append(content.text)
                label = content.text[:100].replace("\n", " ")
                print(f"[{i}] MESSAGE: {label}...")
    elif item.type == "code_interpreter_call":
        print(f"[{i}] CODE_INTERPRETER (status={item.status})")
        print(f"    Container: {item.container_id}")
        code_lines = item.code.strip().split("\n")
        print(f"    Code: {len(code_lines)} lines")
        # Show first and last few lines
        for line in code_lines[:3]:
            print(f"      {line}")
        if len(code_lines) > 6:
            print(f"      ... ({len(code_lines) - 6} more lines)")
        for line in code_lines[-3:]:
            print(f"      {line}")
    print()

# Combine all text messages for the full answer
print("=" * 60)
print("FULL ANALYSIS")
print("=" * 60)
for part in all_text_parts:
    print(part)
    print()

print(f"--- Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out ---")
print(f"--- Output items: {len(response.output)} ---")

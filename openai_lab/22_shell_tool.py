"""Exercise 22: Shell tool — run commands in a hosted container."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: Basic shell command ---
print("=" * 60)
print("EXAMPLE 1: Basic shell commands in a hosted container")
print("=" * 60)
print()
print("The shell tool gives the model a full Linux environment.")
print("Unlike code_interpreter (Python only), shell supports any command.")
print()

response = client.responses.create(
    model="gpt-5.5",
    tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
    input="Check what tools are available in this environment: Python version, Node version, and list /mnt/data contents.",
)

print(f"Response:\n{response.output_text}")

print("\nOutput items:")
for i, item in enumerate(response.output):
    print(f"  [{i}] {item.type}")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# --- Example 2: Data processing with shell tools ---
print()
print("=" * 60)
print("EXAMPLE 2: Data analysis with shell commands")
print("=" * 60)
print()

response2 = client.responses.create(
    model="gpt-5.5",
    tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
    instructions="Use shell commands to complete the task. Show your work.",
    input="""Create a CSV file with this data, then use shell tools to analyze it:

Name,Department,Salary,Start_Date
Alice,Engineering,145000,2021-03-15
Bob,Engineering,132000,2022-07-01
Carol,Sales,98000,2020-11-20
Dave,Engineering,155000,2019-06-10
Eve,Sales,105000,2023-01-15
Frank,Marketing,88000,2022-09-01
Grace,Engineering,140000,2021-08-20
Hank,Marketing,92000,2020-04-05

Tasks:
1. Write the CSV to /mnt/data/employees.csv
2. Use awk to calculate average salary per department
3. Use sort to find the highest paid employee
4. Use grep + wc to count engineering employees
5. Use cut + sort + uniq to list unique departments
""",
)

print(f"Response:\n{response2.output_text}")

print("\nOutput items:")
for i, item in enumerate(response2.output):
    print(f"  [{i}] {item.type}")

print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")

# --- Example 3: Shell + code_interpreter together ---
print()
print("=" * 60)
print("EXAMPLE 3: Shell + code_interpreter in one request")
print("=" * 60)
print()

response3 = client.responses.create(
    model="gpt-5.5",
    tools=[
        {"type": "shell", "environment": {"type": "container_auto"}},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
    input=(
        "Use shell to run 'curl -s https://api.github.com/repos/openai/openai-python' "
        "and capture the JSON response. Then use code_interpreter to parse the JSON "
        "and extract: stars, forks, open issues, and last push date. "
        "Present a clean summary."
    ),
)

print(f"Response:\n{response3.output_text}")

print("\nTool usage chain:")
for i, item in enumerate(response3.output):
    print(f"  [{i}] {item.type}")

print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")

# --- Summary ---
print()
print("=" * 60)
print("SHELL TOOL KEY CONCEPTS")
print("=" * 60)
print("""
Tool config:
  {"type": "shell", "environment": {"type": "container_auto"}}

Shell vs code_interpreter:
  code_interpreter  — Python-only sandbox, great for data analysis & charts
  shell             — Full Linux environment (Python, Node, curl, awk, grep, etc.)

Key points:
  - container_auto: OpenAI manages the container lifecycle
  - Working directory: /mnt/data
  - Response items: shell_call (command) + shell_call_output (result)
  - Can combine with other tools (code_interpreter, web_search, etc.)
  - Ideal for: file processing, system commands, multi-language execution
  - Model: gpt-5.5 (default in shell docs); gpt-5.4 family also supported

Environment options:
  container_auto       — OpenAI provisions and tears down the container
  container_reference  — Reuse an existing container by container_id
  local                — Run commands in your own runtime (you implement the executor)
""")

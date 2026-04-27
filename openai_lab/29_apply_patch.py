"""Exercise 29: Apply patch tool — codex-style file editing.

The `apply_patch` tool lets the model emit structured file operations
(create, update, delete) as V4A-format diffs. Your harness applies them
and reports back. This is what powers Codex and is the right primitive
for any agent that edits real source files.

Three operation types:
  create_file  — V4A diff representing full contents
  update_file  — V4A diff with additions / deletions
  delete_file  — path only

Models: gpt-5.1, 5.2, 5.4, 5.5 (Responses API, Chat Completions, Assistants).

Reference: https://developers.openai.com/api/docs/guides/tools-apply-patch
"""

import os
import shutil
import subprocess
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# === Set up a tiny throwaway repo for the model to edit ===

WORKDIR = tempfile.mkdtemp(prefix="apply_patch_demo_")

INITIAL_FILES = {
    "lib/fib.py": (
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    ),
    "run.py": (
        "from lib.fib import fib\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    for i in range(10):\n"
        "        print(fib(i))\n"
    ),
}

for path, content in INITIAL_FILES.items():
    full = os.path.join(WORKDIR, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)

print(f"=== Working dir: {WORKDIR} ===")
for path in INITIAL_FILES:
    print(f"  {path}")

# === Ask the model to refactor across both files ===
# We pass file contents inline. In a real harness you'd give the model
# read/grep tools and let it explore, then emit patches.

prompt = f"""Rename `fib` to `fibonacci` in both files and add a one-line
docstring to the function. Keep behavior identical.

=== lib/fib.py ===
{INITIAL_FILES['lib/fib.py']}
=== run.py ===
{INITIAL_FILES['run.py']}
"""

response = client.responses.create(
    model="gpt-5.5",
    tools=[{"type": "apply_patch"}],
    input=prompt,
)

# === Apply the patches the model emitted ===

print("\n=== Patches emitted by the model ===")
patch_calls = [item for item in response.output if item.type == "apply_patch_call"]
print(f"Got {len(patch_calls)} apply_patch_call(s)")

apply_patch_outputs = []

for call in patch_calls:
    op = call.operation
    print(f"\n  -> {op.type} {op.path}")
    if op.type in ("create_file", "update_file"):
        # Show the diff
        diff_preview = op.diff[:300] + ("..." if len(op.diff) > 300 else "")
        print(f"     diff:\n{diff_preview}")

    # Apply the operation. In production, do this through your VCS / sandbox.
    target = os.path.join(WORKDIR, op.path)

    try:
        if op.type == "delete_file":
            os.remove(target)
            status = "completed"
            output_msg = ""
        elif op.type == "create_file":
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # V4A diff for create_file represents full contents — extract
            # added lines (lines starting with '+').
            content = "\n".join(
                line[1:] for line in op.diff.splitlines() if line.startswith("+")
            ) + "\n"
            with open(target, "w") as f:
                f.write(content)
            status = "completed"
            output_msg = ""
        elif op.type == "update_file":
            # In production you'd pipe the V4A diff through `apply_patch` /
            # `git apply`. For demo purposes we just write what the diff
            # represents — naive merge of context + added lines.
            new_lines = []
            for line in op.diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    new_lines.append(line[1:])
                # Skip removed lines (start with '-')
            with open(target, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            status = "completed"
            output_msg = ""
        else:
            status = "failed"
            output_msg = f"unknown operation {op.type}"
    except Exception as e:
        status = "failed"
        output_msg = str(e)[:200]

    apply_patch_outputs.append({
        "type": "apply_patch_call_output",
        "call_id": call.call_id,
        "status": status,
        "output": output_msg,
    })

# === Send the results back so the model can confirm and respond ===

if apply_patch_outputs:
    follow_up = client.responses.create(
        model="gpt-5.5",
        tools=[{"type": "apply_patch"}],
        previous_response_id=response.id,
        input=apply_patch_outputs,
    )
    print("\n=== Model's confirmation ===")
    print(follow_up.output_text)

# === Verify the rename actually worked ===

print("\n=== Final file contents ===")
for path in INITIAL_FILES:
    full = os.path.join(WORKDIR, path)
    if os.path.exists(full):
        print(f"\n--- {path} ---")
        with open(full) as f:
            print(f.read())

# Run the renamed code to prove it still works
result = subprocess.run(
    ["python", "run.py"],
    cwd=WORKDIR,
    capture_output=True,
    text=True,
)
print(f"\n=== Running run.py ===")
print(f"stdout: {result.stdout.strip()}")
if result.returncode != 0:
    print(f"stderr: {result.stderr.strip()}")

# Cleanup
shutil.rmtree(WORKDIR)

# === When to use apply_patch ===
print("\n" + "=" * 70)
print("WHEN TO USE apply_patch")
print("=" * 70)
print("""
Use apply_patch when:
  - The agent edits real source files in a repo (Codex-style flows)
  - You want structured, reviewable diffs instead of free-form text edits
  - You need atomic multi-file changes
  - You want the model to commit through your existing VCS pipeline

Don't use apply_patch when:
  - The agent only writes new files in a sandbox (use shell + write directly)
  - Output is meant to be displayed, not committed (use plain text)

Production patterns:
  - Pipe the V4A diff through `git apply --check` first to validate
  - Run tests after applying; if they fail, send the failure back as
    apply_patch_call_output with status=failed and let the model retry
  - Cap the number of patch turns (e.g. 5) to avoid runaway loops
  - Combine with apply_patch + shell tool: shell explores the codebase,
    apply_patch emits the changes
""")

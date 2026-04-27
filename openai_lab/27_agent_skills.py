"""Exercise 27: Agent skills — reusable capability bundles for agents.

A "skill" is a folder bundle that packages a capability for an agent:

    customer_health_review/
        SKILL.md           # required: metadata + instructions
        prompts/
            qbr_template.md
        scripts/
            score_account.py
        examples/
            sample_input.json

The SKILL.md is the entry point — it tells the agent what the skill does,
when to invoke it, what inputs it expects, and what files in the bundle to
reference. Skills compose with the shell tool, function tools, and MCP
servers, and are typically invoked via the Responses API skills config.

Reference: https://developers.openai.com/api/docs/guides/tools-shell
           https://developers.openai.com/blog/skills-shell-tips
"""

import os
import shutil
import textwrap

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# === Step 1: Build a skill bundle on disk ===

SKILL_ROOT = "/tmp/skills/customer_health_review"


def write(path, content):
    full = os.path.join(SKILL_ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)


# Clean slate
if os.path.exists(SKILL_ROOT):
    shutil.rmtree(SKILL_ROOT)

# The SKILL.md is the contract. Keep instructions short and operational.
write("SKILL.md", textwrap.dedent("""\
    ---
    name: customer_health_review
    description: Assess the health of an enterprise CloudSync account and
      produce a QBR-ready summary. Use when the user provides a customer
      name plus recent usage / support / contract data.
    inputs:
      - customer_name (string)
      - usage_summary (string)
      - support_summary (string)
      - contract_summary (string)
    outputs:
      - A markdown QBR brief following prompts/qbr_template.md
    ---

    # Customer Health Review

    1. Read prompts/qbr_template.md and follow that exact structure.
    2. Score the account using scripts/score_account.py logic
       (or invoke the script via the shell tool if available).
    3. Flag any of these critical signals explicitly:
        - Champion change in the last 30 days
        - Usage decline > 20% MoM for 2+ months
        - Open P1 tickets unresolved > 7 days
        - Renewal within 90 days AND health score < 60
    4. End with 3 concrete CSM next-actions, each owned by a named role.
    """))

write("prompts/qbr_template.md", textwrap.dedent("""\
    # QBR — {customer_name}

    ## Health Score: {score}/100  ({tier_label})

    ## What's Working
    - {positive_signal_1}
    - {positive_signal_2}

    ## Risks
    - {risk_1}
    - {risk_2}

    ## Recommended Actions (next 30 days)
    1. {action_1}  — owner: {owner_1}
    2. {action_2}  — owner: {owner_2}
    3. {action_3}  — owner: {owner_3}
    """))

write("scripts/score_account.py", textwrap.dedent("""\
    \"\"\"Score an account 0-100 from raw signals.\"\"\"
    def score(usage_change_pct, p1_open, days_to_renewal, champion_change):
        s = 80
        if usage_change_pct < -20: s -= 25
        if p1_open > 0:            s -= 10 * p1_open
        if days_to_renewal < 90:   s -= 10
        if champion_change:        s -= 15
        return max(0, min(100, s))
    """))

print("=" * 70)
print("STEP 1: Built skill bundle")
print("=" * 70)
for root, _, files in os.walk(SKILL_ROOT):
    rel = os.path.relpath(root, SKILL_ROOT)
    for f in files:
        path = os.path.join(rel, f) if rel != "." else f
        size = os.path.getsize(os.path.join(root, f))
        print(f"  {path:<35} {size:>5} bytes")

# === Step 2: Use the skill via the shell tool ===
#
# The hosted shell can read the skill bundle and follow SKILL.md. In a real
# deployment you'd publish the skill folder to OpenAI (or a workspace mount)
# so agents can `cat SKILL.md` and operate from there.

print("\n" + "=" * 70)
print("STEP 2: Invoke the skill via the shell tool")
print("=" * 70)

skill_text = open(os.path.join(SKILL_ROOT, "SKILL.md")).read()
template_text = open(os.path.join(SKILL_ROOT, "prompts/qbr_template.md")).read()
script_text = open(os.path.join(SKILL_ROOT, "scripts/score_account.py")).read()

# For local execution we inline the skill files. In production you'd mount
# the bundle into the container or push it to the workspace.
prompt = f"""You have access to the `customer_health_review` skill. The
bundle is below.

=== SKILL.md ===
{skill_text}

=== prompts/qbr_template.md ===
{template_text}

=== scripts/score_account.py ===
{script_text}

Apply the skill to this account:

  customer_name:    Meridian Financial Services
  usage_summary:    API calls down 23% MoM for the last 3 months; 47 active users (was 62)
  support_summary:  2 P1 tickets open >10 days (SSO auth failures); 1 P2 (sync delays)
  contract_summary: $480K/yr Enterprise; renews in 73 days; champion (VP Eng) left 14 days ago

Use the score_account function logic to compute the health score.
Output the QBR brief filling in the template exactly.
"""

response = client.responses.create(
    model="gpt-5.5",
    tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
    input=prompt,
)

print(response.output_text)

# === Cleanup ===
shutil.rmtree(SKILL_ROOT)

# === Skills design notes ===
print("\n" + "=" * 70)
print("AGENT SKILL DESIGN NOTES")
print("=" * 70)
print("""
A skill is just a folder + SKILL.md. The discipline is what makes it work:

1. SKILL.md owns the contract.
   Name, description, inputs, outputs, and step-by-step instructions.
   The model treats this as a runbook, not a suggestion.

2. Keep skills SINGLE-PURPOSE.
   "customer_health_review" — yes. "customer_success_toolkit" — no.
   One skill = one reproducible deliverable.

3. Bundle the prompts, templates, and scripts the skill needs.
   If a CSM template is in Confluence and the skill can't read it, the
   skill is broken. Pin everything inside the bundle.

4. Skills compose with tools, not replace them.
   A skill might call the shell tool to run a script, function tools to
   hit your CRM, or MCP servers for external data. SKILL.md tells the
   model which tools to reach for and when.

5. Version skills explicitly.
   Add a `version:` field in SKILL.md frontmatter. Agents that pin a
   version don't break when you ship v2.

6. Pairs naturally with:
   - Shell tool (Exercise 22) — execute scripts in the skill bundle
   - Compaction (Exercise 26) — long-running skill workflows
   - Evals (Exercise 28) — measure that the skill produces stable output
""")

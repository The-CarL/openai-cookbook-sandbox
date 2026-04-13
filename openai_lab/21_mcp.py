"""Exercise 21: MCP (Model Context Protocol) — connect models to external services."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Example 1: Simple MCP server (D&D dice roller) ---
print("=" * 60)
print("EXAMPLE 1: Simple MCP server — D&D dice roller")
print("=" * 60)
print()
print("MCP lets models call tools hosted on remote servers.")
print("The model discovers available tools from the server at runtime.")
print()

response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[
        {
            "type": "mcp",
            "server_label": "dmcp",
            "server_description": "A Dungeons and Dragons MCP server for dice rolling.",
            "server_url": "https://dmcp-server.deno.dev/sse",
            "require_approval": "never",
        },
    ],
    input="Roll 3d6 for my character's strength stat, then roll 1d20 for an attack.",
)

print(f"Response: {response.output_text}")

# Show the output items — MCP calls appear as mcp_call / mcp_call_output pairs
print("\nOutput items:")
for i, item in enumerate(response.output):
    item_type = item.type
    print(f"  [{i}] {item_type}")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# --- Example 2: DeepWiki — query open-source project documentation ---
print()
print("=" * 60)
print("EXAMPLE 2: DeepWiki MCP — query open-source documentation")
print("=" * 60)
print()

response2 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": {
                "never": {
                    "tool_names": ["ask_question", "read_wiki_structure"],
                },
            },
        },
    ],
    input=(
        "Using the deepwiki MCP server, look up the openai/openai-python project "
        "and tell me: what is the Responses API and how does it differ from "
        "Chat Completions?"
    ),
)

print(f"Response: {response2.output_text[:600]}")
if len(response2.output_text) > 600:
    print("...")

print(f"\nOutput items:")
for i, item in enumerate(response2.output):
    print(f"  [{i}] {item.type}")

print(f"\nTokens: {response2.usage.input_tokens} in, {response2.usage.output_tokens} out")

# --- Example 3: Multiple MCP servers in one call ---
print()
print("=" * 60)
print("EXAMPLE 3: Multiple MCP servers + built-in tools together")
print("=" * 60)
print()

response3 = client.responses.create(
    model="gpt-4.1-mini",
    tools=[
        # MCP server
        {
            "type": "mcp",
            "server_label": "dmcp",
            "server_description": "D&D dice roller",
            "server_url": "https://dmcp-server.deno.dev/sse",
            "require_approval": "never",
        },
        # Built-in tool alongside MCP
        {"type": "web_search"},
    ],
    input=(
        "I'm building a D&D character. Roll 4d6 drop lowest for each of my "
        "6 ability scores. Also search the web for the standard D&D 5e point buy costs "
        "so I can compare which method gives me a better character."
    ),
)

print(f"Response: {response3.output_text[:600]}")
if len(response3.output_text) > 600:
    print("...")

print(f"\nTool usage chain:")
for i, item in enumerate(response3.output):
    print(f"  [{i}] {item.type}")

print(f"\nTokens: {response3.usage.input_tokens} in, {response3.usage.output_tokens} out")

# --- Summary ---
print()
print("=" * 60)
print("MCP KEY CONCEPTS")
print("=" * 60)
print("""
MCP (Model Context Protocol) connects models to remote tool servers.

Tool config:
  {
      "type": "mcp",
      "server_label": "my_server",          # Your label for this server
      "server_url": "https://...",           # The MCP server endpoint
      "server_description": "...",           # Optional: helps the model understand when to use it
      "require_approval": "never"            # Or per-tool approval policies
  }

Approval policies:
  "never"                          — auto-approve all tools on this server
  {"never": {"tool_names": [...]}} — auto-approve specific tools only
  "always"                         — require approval for every call (default)

Key points:
  - The model discovers available tools from the MCP server at runtime
  - You can mix MCP servers with built-in tools (web_search, etc.) and functions
  - Multiple MCP servers can be used in a single request
  - MCP servers must implement the Model Context Protocol spec
  - Public servers: deepwiki, Stripe, Cloudflare, and many more
""")

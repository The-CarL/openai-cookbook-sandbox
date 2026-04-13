"""Exercise 23: Computer use — model operates software via screenshots and actions."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# --- Overview ---
print("=" * 60)
print("COMPUTER USE — GPT-5.4 operates GUIs via screenshots")
print("=" * 60)
print()
print("Computer use lets the model interact with software by:")
print("  1. Observing the screen (via screenshots you provide)")
print("  2. Returning structured actions (click, type, scroll, drag)")
print("  3. You execute the actions in a browser/VM and send back a new screenshot")
print("  4. Loop until the task is complete")
print()

# --- Example 1: Initial API call showing the tool setup ---
print("=" * 60)
print("EXAMPLE 1: Computer use API call structure")
print("=" * 60)
print()

# The computer tool requires a screenshot to act on.
# On the first call, the model will request a screenshot.
response = client.responses.create(
    model="gpt-5.4",
    tools=[{"type": "computer"}],
    input="Go to openai.com and find the pricing page for the API.",
    # truncation="auto" is recommended for computer use loops
    truncation="auto",
)

print("Output items from initial call:")
for i, item in enumerate(response.output):
    item_type = item.type
    print(f"  [{i}] type={item_type}")

    # computer_call items contain the actions the model wants to execute
    if item_type == "computer_call":
        print(f"       call_id={item.call_id}")
        # The action tells us what the model wants to do
        action = item.action
        print(f"       action type={action.type}")
        if action.type == "screenshot":
            print("       -> Model is requesting a screenshot of the current screen")
        elif action.type == "click":
            print(f"       -> Click at ({action.x}, {action.y}) button={action.button}")
        elif action.type == "type":
            print(f"       -> Type: '{action.text}'")
        elif action.type == "scroll":
            print(f"       -> Scroll delta=({action.delta_x}, {action.delta_y})")

print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# --- Example 2: The full computer use loop pattern ---
print()
print("=" * 60)
print("THE COMPUTER USE LOOP (reference pattern)")
print("=" * 60)
print("""
A complete implementation requires a browser/VM harness. Here's the pattern:

```python
from openai import OpenAI
import base64

client = OpenAI()

# Step 1: Start the task
response = client.responses.create(
    model="gpt-5.4",
    tools=[{"type": "computer"}],
    input="Navigate to example.com and fill in the contact form.",
    truncation="auto",
)

MAX_TURNS = 20

for turn in range(MAX_TURNS):
    # Step 2: Find computer_call items in the response
    computer_calls = [item for item in response.output if item.type == "computer_call"]

    if not computer_calls:
        break  # Model is done — no more actions requested

    results = []
    for call in computer_calls:
        action = call.action

        if action.type == "screenshot":
            # Take a screenshot of the current screen
            screenshot_b64 = take_screenshot()  # Your harness function
        elif action.type == "click":
            execute_click(action.x, action.y, action.button)
            screenshot_b64 = take_screenshot()
        elif action.type == "type":
            execute_type(action.text)
            screenshot_b64 = take_screenshot()
        elif action.type == "scroll":
            execute_scroll(action.delta_x, action.delta_y)
            screenshot_b64 = take_screenshot()
        elif action.type == "drag":
            execute_drag(action.x, action.y, action.x2, action.y2)
            screenshot_b64 = take_screenshot()

        # Send the screenshot back as the tool output
        results.append({
            "type": "computer_call_output",
            "call_id": call.call_id,
            "output": {
                "type": "computer_screenshot",
                "image_url": f"data:image/png;base64,{screenshot_b64}",
            },
        })

    # Step 3: Continue the loop with the screenshot results
    response = client.responses.create(
        model="gpt-5.4",
        tools=[{"type": "computer"}],
        previous_response_id=response.id,
        input=results,
        truncation="auto",
    )

# Step 4: Get the final text output
print(response.output_text)
```

Action types returned by the model:
  screenshot  — Request a screenshot (always the first action)
  click       — Click at (x, y) with button (left/right/middle)
  type        — Type text string
  scroll      — Scroll by (delta_x, delta_y) pixels
  drag        — Drag from (x, y) to (x2, y2)

Key points:
  - GA tool type: {"type": "computer"} with model gpt-5.4
  - (Preview used: {"type": "computer_use_preview"} with model computer-use-preview)
  - Always use truncation="auto" to manage growing context from screenshots
  - Run in an isolated environment (VM, container, sandboxed browser)
  - Maintain human oversight for high-impact actions (purchases, form submissions)
  - Screenshot detail: use original resolution for best results
  - Ideal for: form filling, web navigation, UI testing, desktop automation
""")

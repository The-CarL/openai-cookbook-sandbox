"""Exercise 32: gpt-realtime-2 — GPT-5-class reasoning in the Realtime API.

Released May 7, 2026. Three new Realtime API voice models:
  gpt-realtime-2         — speech-to-speech with configurable reasoning (128K ctx)
  gpt-realtime-translate — live speech translation (70+ input, 13 output languages)
  gpt-realtime-whisper   — streaming speech-to-text transcription

Requires: uv sync  (openai[realtime] pulls websockets>=13)
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()

# Three new Realtime API models (May 7, 2026)
REALTIME_MODELS = {
    "gpt-realtime-2": {
        "desc": "GPT-5-class reasoning voice model. Default for production voice agents.",
        "context": "128K tokens (4x gpt-realtime-1.5)",
        "pricing": "$32/M audio-input, $64/M audio-output, $0.40/M cached",
        "reasoning": "minimal / low / medium / high / xhigh  (default: low)",
    },
    "gpt-realtime-translate": {
        "desc": "Live speech-to-speech translation. 70+ input, 13 output languages.",
        "pricing": "$0.034/min",
    },
    "gpt-realtime-whisper": {
        "desc": "Streaming speech-to-text transcription. Low-latency transcription pipeline.",
        "pricing": "$0.017/min",
    },
}


async def example_1_session_init():
    """Connect to gpt-realtime-2 and configure the session."""
    print("=" * 60)
    print("EXAMPLE 1: Session initialization with gpt-realtime-2")
    print("=" * 60)
    print()

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        # Server sends session.created immediately on connect
        created = await conn.recv()
        assert created.type == "session.created", f"Unexpected first event: {created.type}"
        print(f"Model:      {created.session.model}")
        print(f"Session ID: {created.session.id}")

        # Configure text-only mode + reasoning effort.
        # The 'reasoning' key is new in gpt-realtime-2 and not yet in the
        # SDK's typed session params, so we use conn.send() with a raw dict.
        await conn.send({
            "type": "session.update",
            "session": {
                "output_modalities": ["text"],
                "reasoning": {"effort": "medium"},
                "instructions": "You are a concise assistant. Keep answers to 2-3 sentences.",
            },
        })

        updated = await conn.recv()
        assert updated.type == "session.updated", f"Unexpected event: {updated.type}"
        print(f"\nSession updated.")
        print(f"Output modalities: {updated.session.output_modalities}")
        # reasoning is not yet in the typed Session model; inspect via raw dict
        raw = updated.model_dump()
        effort = raw.get("session", {}).get("reasoning", {}).get("effort", "unknown")
        print(f"Reasoning effort:  {effort}")


async def example_2_text_reasoning():
    """Send a text query and stream the response."""
    print()
    print("=" * 60)
    print("EXAMPLE 2: Text conversation with reasoning_effort=low")
    print("=" * 60)
    print()

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.recv()  # session.created

        # Low reasoning effort: fastest, good for most voice agent workflows
        await conn.send({
            "type": "session.update",
            "session": {
                "output_modalities": ["text"],
                "reasoning": {"effort": "low"},
            },
        })
        await conn.recv()  # session.updated

        # Add a user message as a conversation item
        conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What are two advantages of GPT-5-class reasoning in a voice agent?"}],
        })
        ev = await conn.recv()
        assert ev.type == "conversation.item.created", f"Unexpected event: {ev.type}"

        # Request a response
        conn.response.create()

        # Stream response events until response.done
        response_text = ""
        input_tokens = output_tokens = 0
        while True:
            event = await conn.recv()
            if event.type == "response.text.delta":
                response_text += event.delta
            elif event.type == "response.done":
                usage = getattr(event.response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                break
            elif event.type == "error":
                print(f"Error: {event.error}")
                return

        print(f"Response:\n{response_text}")
        if input_tokens or output_tokens:
            print(f"\nTokens: {input_tokens} input, {output_tokens} output")


async def main():
    await example_1_session_init()
    await example_2_text_reasoning()

    # Model summary
    print()
    print("=" * 60)
    print("REALTIME MODEL LINEUP (May 7, 2026)")
    print("=" * 60)
    for name, info in REALTIME_MODELS.items():
        print(f"\n{name}")
        for k, v in info.items():
            print(f"  {k:<10}: {v}")

    print("""
AUDIO CONVERSATION LOOP (reference — requires mic/speaker harness):

```python
from openai import AsyncOpenAI
import asyncio, base64

client = AsyncOpenAI()

async def voice_agent():
    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.recv()  # session.created
        await conn.send({
            "type": "session.update",
            "session": {
                "output_modalities": ["audio", "text"],
                "reasoning": {"effort": "low"},      # raise for complex tasks
                "turn_detection": {"type": "server_vad"},
            },
        })
        await conn.recv()  # session.updated

        # Stream audio from microphone (PCM16, 24 kHz, mono, base64-encoded)
        while capturing:
            chunk = mic.read(4096)
            await conn.send({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            })

        # Model responds via response.audio.delta events
        async for event in conn:
            if event.type == "response.audio.delta":
                speaker.write(base64.b64decode(event.delta))
            elif event.type == "response.done":
                break
```

Reasoning effort levels for gpt-realtime-2:
  minimal  — fastest; trivial commands and FAQ routing
  low      — default; most production voice agents
  medium   — complex multi-step question answering
  high     — code debugging, math, structured analysis
  xhigh    — hardest tasks; highest latency, highest quality

Upgrade path from gpt-realtime-1.5:
  Context:  32K → 128K tokens (4x)
  Reasoning: none → 5 configurable effort levels
  Tool use:  sequential → parallel (with spoken preambles)
  Transport: same WebSocket / WebRTC endpoints

  ⚠ Realtime API Beta was shut down May 12, 2026.
    If you used the Beta endpoint, migrate to the GA Realtime API.

gpt-realtime-translate: WebSocket, send PCM16 audio, receive translated audio
gpt-realtime-whisper:   WebSocket, send PCM16 audio, receive streaming text transcript
""")


if __name__ == "__main__":
    asyncio.run(main())

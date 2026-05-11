"""Exercise 32: Realtime API — gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

Three new Realtime API models released May 7, 2026. Realtime API is now GA.

  gpt-realtime-2          Speech-to-speech with GPT-5-class reasoning.
                          $32 / 1M audio input, $64 / 1M audio output.
  gpt-realtime-translate  Live speech translation (70+ input → 13 output languages).
                          $0.034 / min.
  gpt-realtime-whisper    Streaming speech-to-text transcription.
                          $0.017 / min.

The Realtime API uses WebSockets — the Python SDK wraps this via
`client.realtime.connect()`. This exercise runs text mode (no microphone
needed) to demonstrate the protocol; the audio patterns are shown as
reference code.

Requires: openai>=2.1.0 (included in this project)
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()


# --- Example 1: gpt-realtime-2 in text mode (runnable without audio hardware) ---

async def text_mode_demo() -> str:
    print("=" * 60)
    print("EXAMPLE 1: gpt-realtime-2 in text mode")
    print("=" * 60)
    print()
    print("Text modality — no microphone or speaker required.")
    print("In production, swap input_text → audio buffer append,")
    print("and listen for audio.delta events instead of text delta.")
    print()

    full_text = ""

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        # Configure session for text-only I/O.
        # reasoning_effort controls the depth of in-call reasoning:
        #   "low"    fastest, lightest (≈ Instant mode)
        #   "medium" default
        #   "high"   deepest, slowest
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": "You are a customer success assistant. Answer concisely.",
            "reasoning_effort": "low",
        })

        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the Responses API in one sentence?"}],
        })

        await conn.response.create()

        async for event in conn:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
                full_text += event.delta
            elif event.type == "response.output_text.done":
                print()
            elif event.type == "response.done":
                usage = getattr(event.response, "usage", None)
                if usage:
                    print(f"\nUsage: {usage.input_tokens} in, {usage.output_tokens} out")
                break

    return full_text


# --- Example 2: Multi-turn text conversation ---

async def multi_turn_demo():
    print()
    print("=" * 60)
    print("EXAMPLE 2: Multi-turn conversation with gpt-realtime-2")
    print("=" * 60)
    print()

    questions = [
        "What is prompt caching?",
        "How does that affect pricing?",
    ]

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["text"],
            "instructions": "You are a helpful API expert. Keep answers under 40 words.",
        })

        for q in questions:
            print(f"User: {q}")
            print("Model: ", end="")

            await conn.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": q}],
            })
            await conn.response.create()

            async for event in conn:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.output_text.done":
                    print()
                elif event.type == "response.done":
                    break

            print()


# --- Reference patterns (audio — requires microphone / speaker harness) ---

AUDIO_PATTERN = """
gpt-realtime-2 AUDIO MODE (speech-to-speech):

```python
import base64, asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def voice_agent():
    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "semantic_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "marin",   # New voice added with gpt-realtime-2
                },
            },
            "instructions": "You are a helpful voice assistant.",
            "reasoning_effort": "medium",
        })

        # Stream PCM audio from microphone in chunks
        for chunk in microphone_stream():          # your capture loop
            await conn.input_audio_buffer.append(
                audio=base64.b64encode(chunk).decode()
            )

        # Receive events
        async for event in conn:
            if event.type == "response.audio.delta":
                play_audio(base64.b64decode(event.delta))  # your speaker
            elif event.type == "response.audio_transcript.delta":
                print(event.delta, end="")   # live transcript of model speech
            elif event.type == "response.done":
                break
```
"""

TRANSLATE_PATTERN = """
gpt-realtime-translate (live speech translation, 70+ → 13 languages):

```python
async with client.realtime.connect(model="gpt-realtime-translate") as conn:
    await conn.session.update(session={
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {"type": "semantic_vad"},
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "language": "en",   # Target language (one of 13 supported)
            },
        },
    })
    # Stream audio in, receive translated audio/text out.
    # Priced at $0.034/min — NOT per token.
```
"""

WHISPER_PATTERN = """
gpt-realtime-whisper (streaming speech-to-text, $0.017/min):

```python
async with client.realtime.connect(model="gpt-realtime-whisper") as conn:
    await conn.session.update(session={
        "type": "realtime",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {"type": "semantic_vad"},
            },
        },
    })
    for chunk in microphone_stream():
        await conn.input_audio_buffer.append(
            audio=base64.b64encode(chunk).decode()
        )
    async for event in conn:
        if event.type == "conversation.item.input_audio_transcription.completed":
            print(f"Transcript: {event.transcript}")
        elif event.type == "response.done":
            break
```
"""


async def main():
    await text_mode_demo()
    await multi_turn_demo()

    print("=" * 60)
    print("AUDIO MODE PATTERN (requires microphone harness)")
    print("=" * 60)
    print(AUDIO_PATTERN)

    print("=" * 60)
    print("REALTIME TRANSLATE PATTERN")
    print("=" * 60)
    print(TRANSLATE_PATTERN)

    print("=" * 60)
    print("REALTIME WHISPER PATTERN")
    print("=" * 60)
    print(WHISPER_PATTERN)

    print("=" * 60)
    print("KEY CONCEPTS")
    print("=" * 60)
    print("""
Model selection:
  gpt-realtime-2          Voice agent with GPT-5-class reasoning.
                          Use when: low latency + intelligence matter most.
                          Pricing: $32 / 1M audio in, $64 / 1M audio out.

  gpt-realtime-translate  Live speech-to-speech translation (70+ → 13 langs).
                          Use when: multilingual users.
                          Pricing: $0.034 / min (not per token).

  gpt-realtime-whisper    Streaming transcription only (text out).
                          Use when: cheapest real-time audio input.
                          Pricing: $0.017 / min.

GA status (May 7, 2026):
  The Realtime API exited beta — safe for production use.
  WebSocket endpoint: wss://api.openai.com/v1/realtime?model=<model>

Realtime vs Responses API:
  Responses API   HTTP request/response, full tool support, text/image/audio I/O.
  Realtime API    WebSocket, streaming, speech-to-speech, sub-200ms latency.

reasoning_effort (gpt-realtime-2 only):
  "low"    Faster, lighter — equivalent to Instant mode.
  "medium" Default.
  "high"   Deeper reasoning, higher latency and cost.

Turn detection:
  semantic_vad   Model detects natural speech pauses (recommended).
  server_vad     Server-side energy-based VAD (legacy).
""")


if __name__ == "__main__":
    asyncio.run(main())

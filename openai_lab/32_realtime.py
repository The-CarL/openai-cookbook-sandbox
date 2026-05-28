"""Exercise 32: Realtime voice API — gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

Three new GA models released May 7, 2026. gpt-realtime-1.5 deprecated May 12.
Audio demos require microphone hardware; this script uses text-only events.

Requires: pip install openai
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI()


async def example_1_text_session():
    print("=" * 60)
    print("EXAMPLE 1: gpt-realtime-2 via text events (no microphone needed)")
    print("=" * 60)
    print()

    async with client.beta.realtime.connect(model="gpt-realtime-2") as conn:
        # Text-only session; reasoning.effort trades latency vs quality
        await conn.session.update(session={
            "modalities": ["text"],
            "instructions": "Be concise.",
            "reasoning": {"effort": "low"},
        })

        # Send a text message and request a response
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What are the three realtime models OpenAI released in May 2026?"}]
        })
        await conn.response.create()

        print("Response: ", end="", flush=True)
        async for event in conn:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                print("\n")
                break
            elif event.type == "error":
                print(f"\nError: {event.error.message}")
                break

    print("Session closed.\n")


async def main():
    await example_1_text_session()

    # --- Audio loop reference pattern (requires microphone) ---
    print("=" * 60)
    print("AUDIO LOOP PATTERN (gpt-realtime-2) — reference, needs microphone")
    print("=" * 60)
    print("""
```python
import base64, asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def voice_session():
    async with client.beta.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.session.update(session={
            "modalities": ["audio", "text"],
            "voice": "alloy",                     # alloy / echo / fable / onyx / nova / shimmer
            "instructions": "You are a helpful assistant.",
            "input_audio_format": "pcm16",         # PCM16 24kHz mono, base64-encoded
            "output_audio_format": "pcm16",
            "reasoning": {"effort": "medium"},     # "low" / "medium" / "high"
            "turn_detection": {
                "type": "server_vad",              # voice activity detection
                "threshold": 0.5,
                "silence_duration_ms": 500,
            },
        })

        # Append audio from your microphone capture loop
        await conn.input_audio_buffer.append(audio=your_base64_pcm16_chunk)

        async for event in conn:
            if event.type == "response.audio.delta":
                play_audio(base64.b64decode(event.delta))   # your speaker output
            elif event.type == "response.audio_transcript.delta":
                print(event.delta, end="")                  # live transcript
            elif event.type == "response.done":
                break

asyncio.run(voice_session())
```
""")

    print("=" * 60)
    print("gpt-realtime-translate + gpt-realtime-whisper patterns")
    print("=" * 60)
    print("""
gpt-realtime-translate  Live speech translation across 70+ languages.
  Pricing: $0.034 / minute of audio

  async with client.beta.realtime.connect(model="gpt-realtime-translate") as conn:
      await conn.session.update(session={
          "modalities": ["audio"],
          "source_language": "es",       # ISO 639-1 input language
          "target_language": "en",       # ISO 639-1 output language
          "voice": "alloy",
      })

gpt-realtime-whisper    Streaming speech-to-text only (no generation).
  Pricing: $0.017 / minute of audio

  async with client.beta.realtime.connect(model="gpt-realtime-whisper") as conn:
      await conn.session.update(session={
          "modalities": ["text"],        # text transcript output only
          "language": "en",
      })
""")

    print("=" * 60)
    print("KEY CONCEPTS — Realtime API (May 2026)")
    print("=" * 60)
    print("""
Model lineup:
  gpt-realtime-2          Speech-to-speech with GPT-5-class reasoning
                          $32.00 / M audio input tokens ($0.40 cached)
                          $64.00 / M audio output tokens
                          Configurable reasoning effort: low / medium / high
                          Stronger tool use, interruption handling, large context window

  gpt-realtime-translate  Live speech translation, 70+ languages
                          $0.034 / minute of audio

  gpt-realtime-whisper    Streaming speech-to-text only
                          $0.017 / minute of audio

Connection:
  Endpoint: wss://api.openai.com/v1/realtime?model=<model-id>
  SDK:      AsyncOpenAI().beta.realtime.connect(model=...)
  Beta header removed: the GA interface is now the default (no 'OpenAI-Beta: realtime=v1')

Reasoning (gpt-realtime-2 only):
  low    fastest, good for most production voice agents
  medium default for complex instructions or tool use
  high   deepest reasoning, highest latency

Deprecations (May 12, 2026):
  gpt-realtime-1.5        Removed — migrate to gpt-realtime-2
  Realtime API Beta       Remove 'OpenAI-Beta: realtime=v1' header if you had it
""")


if __name__ == "__main__":
    asyncio.run(main())

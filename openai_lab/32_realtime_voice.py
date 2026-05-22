"""Exercise 32: gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

Three new realtime voice models released May 7, 2026. They replace the deprecated
Realtime API Beta (gpt-realtime-1.5, removed May 12, 2026).

  gpt-realtime-2         — GPT-5-class reasoning speech-to-speech, 128K context
  gpt-realtime-translate — Live speech translation, 70+ input → 13 output languages
  gpt-realtime-whisper   — Streaming speech-to-text, lowest-latency transcription

The SDK interface is identical to the old Beta — only the model= parameter changes.
Text-only examples below run without audio hardware; swap modalities to
["text", "audio"] in production.

Pricing (verified May 2026):
  gpt-realtime-2:        $32/1M audio in, $0.40/1M cached, $64/1M audio out
  gpt-realtime-translate: $0.034/min
  gpt-realtime-whisper:   $0.017/min
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()


# --- Example 1: gpt-realtime-2, text-only mode (no audio hardware needed) ---

async def example_1():
    print("=" * 60)
    print("EXAMPLE 1: gpt-realtime-2 — text-only mode")
    print("=" * 60)
    print()
    print("Connect to gpt-realtime-2 and exchange text in real time.")
    print('In production: set modalities=["text", "audio"] + voice/format config.')
    print()

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.session.update(session={
            "modalities": ["text"],
            "instructions": "You are a concise technical assistant. Keep answers under 60 words.",
        })

        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the difference between latency and throughput in distributed systems?"}],
        })

        await conn.response.create()

        full_text = ""
        async for event in conn:
            if event.type == "response.output_text.delta":
                full_text += event.delta
                print(event.delta, end="", flush=True)
            elif event.type in ("response.output_text.done", "response.done"):
                print()
                break

    print(f"\nResponse: {len(full_text)} chars")
    print()


# --- Example 2: Configuring reasoning effort ---

async def example_2():
    print("=" * 60)
    print("EXAMPLE 2: gpt-realtime-2 — reasoning effort levels")
    print("=" * 60)
    print()
    print("Five levels: minimal | low (default) | medium | high | xhigh")
    print("low optimizes for voice latency; use medium/high for complex tool use.")
    print()

    for effort in ["low", "high"]:
        print(f"--- reasoning_effort={effort} ---")
        async with client.realtime.connect(model="gpt-realtime-2") as conn:
            await conn.session.update(session={
                "modalities": ["text"],
                "reasoning_effort": effort,
            })

            await conn.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Count the distinct letters in 'strawberry'."}],
            })

            await conn.response.create()

            async for event in conn:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type in ("response.output_text.done", "response.done"):
                    print()
                    break
        print()


# --- Example 3: gpt-realtime-translate reference pattern ---

def show_translate_pattern():
    print("=" * 60)
    print("EXAMPLE 3: gpt-realtime-translate — reference pattern")
    print("=" * 60)
    print()
    print("Streams translated speech in real time.")
    print("70+ input languages, 13 output languages, $0.034/min.")
    print()
    print("""
async with client.realtime.connect(model="gpt-realtime-translate") as conn:
    await conn.session.update(session={
        "modalities": ["audio"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "translation": {
            "input_language": "es",    # source (ISO 639-1)
            "output_language": "en",   # target
        },
    })

    # Stream mic audio → send as input_audio_buffer.append events
    # Receive translated speech as response.output_audio.delta events
    # No intermediate transcript: input speech → output speech directly
""")


# --- Run all examples and print key concepts ---

async def main():
    await example_1()
    await example_2()
    show_translate_pattern()

    print("=" * 60)
    print("KEY CONCEPTS — REALTIME VOICE MODELS (May 2026)")
    print("=" * 60)
    print("""
Model summary:

  gpt-realtime-2
    Context:   128K tokens (4x the 32K of gpt-realtime-1.5)
    Reasoning: minimal / low* / medium / high / xhigh
    Pricing:   $32/1M audio in | $0.40/1M cached | $64/1M audio out
    Use for:   Speech-to-speech agents requiring GPT-5-class reasoning

  gpt-realtime-translate
    Pricing:   $0.034/min
    Use for:   Real-time interpreting, multilingual customer service

  gpt-realtime-whisper
    Pricing:   $0.017/min
    Use for:   Live captions, voice-to-text pipelines (lower latency than batch Whisper)

Migration from gpt-realtime-1.5 (deprecated May 12, 2026):
  Change model="gpt-realtime-1.5" → model="gpt-realtime-2"
  Optionally set reasoning_effort (was absent in 1.5).
  Context window expands 4x automatically; no code changes required.

Full audio loop pattern (requires mic/speaker harness):

  async with client.realtime.connect(model="gpt-realtime-2") as conn:
      await conn.session.update(session={
          "modalities": ["text", "audio"],
          "voice": "shimmer",
          "input_audio_format": "pcm16",
          "output_audio_format": "pcm16",
          "turn_detection": {"type": "server_vad"},
          "reasoning_effort": "medium",
      })

      # Stream mic audio → input_audio_buffer.append events
      # response.output_audio.delta          → speaker playback chunks
      # response.output_audio_transcript.delta → live caption text
""")


if __name__ == "__main__":
    asyncio.run(main())

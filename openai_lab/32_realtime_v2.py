"""Exercise 32: Realtime API v2 — gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

Released May 7, 2026. Three new WebSocket-based voice models replacing the beta gpt-realtime-1.5.
The Realtime API Beta was deprecated May 12, 2026; migrate to the GA interface shown here.

Full voice I/O requires additional audio libraries (pyaudio, sounddevice).
This exercise uses text-only mode so it runs without a microphone.

Requires openai >= 2.1.0:  uv add "openai>=2.1.0"
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()


# ---------------------------------------------------------------------------
# Example 1: gpt-realtime-2 — reasoning voice model, text-only mode
# ---------------------------------------------------------------------------

async def example_1_realtime2_text():
    """Connect with gpt-realtime-2 using text input/output (no audio hardware needed)."""
    print("=" * 60)
    print("EXAMPLE 1: gpt-realtime-2 — text-only mode")
    print("=" * 60)
    print()

    async with client.realtime.connect(model="gpt-realtime-2") as conn:
        # Configure session: text in, text out, medium reasoning
        await conn.session.update(session={
            "modalities": ["text"],
            "reasoning": {"effort": "medium"},  # minimal/low/medium/high/xhigh
            "instructions": "You are a concise assistant. Answer in 1-2 sentences.",
        })

        # Send a text message
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the Responses API in one sentence?"}],
        })
        await conn.response.create()

        # Stream the response
        full_text = ""
        async for event in conn:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
                full_text += event.delta
            elif event.type == "response.text.done":
                print()  # newline after streaming
            elif event.type == "response.done":
                usage = event.response.usage
                if usage:
                    print(f"\nTokens: {usage.input_tokens} in, {usage.output_tokens} out")
                break
            elif event.type == "error":
                print(f"\nError: {event.error.message}")
                break

    print()


# ---------------------------------------------------------------------------
# Example 2: Reasoning effort — gpt-realtime-2 with higher reasoning
# ---------------------------------------------------------------------------

async def example_2_reasoning_effort():
    """Show how reasoning effort affects quality on a harder question."""
    print("=" * 60)
    print("EXAMPLE 2: Reasoning effort levels (gpt-realtime-2)")
    print("=" * 60)
    print()

    QUESTION = "Why does prompt caching lower latency even for cache misses?"

    for effort in ("low", "high"):
        print(f"--- effort={effort} ---")
        async with client.realtime.connect(model="gpt-realtime-2") as conn:
            await conn.session.update(session={
                "modalities": ["text"],
                "reasoning": {"effort": effort},
            })
            await conn.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": QUESTION}],
            })
            await conn.response.create()

            async for event in conn:
                if event.type == "response.text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.text.done":
                    print()
                elif event.type == "response.done":
                    break
                elif event.type == "error":
                    print(f"\nError: {event.error.message}")
                    break
        print()


# ---------------------------------------------------------------------------
# Example 3: gpt-realtime-translate — live speech translation (reference pattern)
# ---------------------------------------------------------------------------

def example_3_translate_pattern():
    """Print the gpt-realtime-translate connection pattern (requires audio hardware to run)."""
    print("=" * 60)
    print("EXAMPLE 3: gpt-realtime-translate — live speech translation (pattern)")
    print("=" * 60)
    print("""
gpt-realtime-translate: 70+ input languages → 13 output languages.
Priced per minute ($0.034/min), not per token.

```python
async with client.realtime.connect(model="gpt-realtime-translate") as conn:
    await conn.session.update(session={
        "modalities": ["audio"],
        "input_audio_format": "pcm16",   # 24kHz 16-bit PCM
        "output_audio_format": "pcm16",
        "source_language": "es",          # input language (ISO 639-1)
        "target_language": "en",          # output language
    })

    # Stream mic audio as input_audio_buffer.append events, then:
    await conn.input_audio_buffer.commit()
    await conn.response.create()

    async for event in conn:
        if event.type == "response.audio.delta":
            # Write event.delta (base64 PCM) to your audio output
            pass
        elif event.type == "response.done":
            break
```

Supported output languages (13): en, es, fr, de, pt, it, nl, pl, sv, da, fi, ru, zh
Input supports 70+ languages (all Whisper-v4 languages).
""")


# ---------------------------------------------------------------------------
# Example 4: gpt-realtime-whisper — streaming STT (reference pattern)
# ---------------------------------------------------------------------------

def example_4_whisper_pattern():
    """Print the gpt-realtime-whisper (streaming STT) connection pattern."""
    print("=" * 60)
    print("EXAMPLE 4: gpt-realtime-whisper — streaming speech-to-text (pattern)")
    print("=" * 60)
    print("""
gpt-realtime-whisper: transcribes speech live as the speaker talks.
Priced per minute ($0.017/min).

```python
async with client.realtime.connect(model="gpt-realtime-whisper") as conn:
    await conn.session.update(session={
        "modalities": ["text"],          # text output only
        "input_audio_format": "pcm16",
        "input_language": "en",          # optional: auto-detect if omitted
    })

    # Feed mic audio in chunks, then commit:
    await conn.input_audio_buffer.append(audio=base64_pcm_chunk)
    await conn.input_audio_buffer.commit()
    await conn.response.create()

    async for event in conn:
        if event.type == "response.text.delta":
            print(event.delta, end="", flush=True)  # live transcript
        elif event.type == "response.done":
            break
```
""")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("=" * 60)
    print("REALTIME API V2 — KEY CONCEPTS")
    print("=" * 60)
    print("""
Three models released May 7, 2026 (Realtime API Beta deprecated May 12):

  gpt-realtime-2          Reasoning voice model. GPT-5-class intelligence,
                          128K context (up from 32K), configurable reasoning
                          effort. Priced per audio token.
                          Input:  $32/1M audio tokens  ($0.40 cached)
                          Output: $64/1M audio tokens

  gpt-realtime-translate  Live speech-to-speech translation.
                          70+ input languages → 13 output languages.
                          Priced: $0.034/min

  gpt-realtime-whisper    Streaming speech-to-text (not speech-to-speech).
                          Live transcript as speaker talks.
                          Priced: $0.017/min

Session config (key fields):
  modalities            ["audio"] for voice, ["text"] for text-only
  reasoning.effort      minimal/low/medium/high/xhigh (gpt-realtime-2 only)
  source_language       input language ISO code (translate/whisper)
  target_language       output language ISO code (translate only)

vs gpt-realtime-1.5 (beta, deprecated):
  - Context: 32K → 128K
  - Reasoning: none → configurable effort levels
  - Parallel tool calls supported
  - Preambles (audible "thinking out loud") configurable

SDK: from openai import AsyncOpenAI
     async with client.realtime.connect(model=...) as conn: ...
     Requires openai >= 2.1.0
""")


async def main():
    await example_1_realtime2_text()
    await example_2_reasoning_effort()
    example_3_translate_pattern()
    example_4_whisper_pattern()
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())

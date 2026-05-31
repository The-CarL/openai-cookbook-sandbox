"""Exercise 32: Realtime Voice API — gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

Three new models released May 7, 2026 into the GA Realtime API:
  gpt-realtime-2          — voice agent with GPT-5-class reasoning
  gpt-realtime-translate  — live speech translation (70+ in → 13 out)
  gpt-realtime-whisper    — streaming speech-to-text

The Beta Realtime API (OpenAI-Beta: realtime=v1 header) was removed May 12, 2026.
All new code should use the GA interface shown here.

Example 1 is fully runnable (text mode, no audio hardware needed).
Example 2 shows the audio-mode session config pattern (requires mic/speaker).
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()

# --- Model overview ---
print("=" * 60)
print("REALTIME VOICE API — Three new models (May 7, 2026)")
print("=" * 60)
print("""
Model                    Use case                               Pricing
──────────────────────────────────────────────────────────────────────────
gpt-realtime-2           Voice agents: reason, call tools,      $32 / M audio-in
                         handle interruptions. GPT-5-class       $64 / M audio-out
                         reasoning. Adjustable effort.           ($0.40 / M cached)
                         128 K context.

gpt-realtime-translate   Live speech translation.                $0.034 / min
                         Input: 70+ languages.
                         Output: 13 languages.

gpt-realtime-whisper     Streaming speech-to-text.               $0.017 / min
                         Transcribes as you speak for
                         low-latency live products.

Migration note
  Beta API header "OpenAI-Beta: realtime=v1" was removed May 12, 2026.
  Drop that header and adopt the GA session shapes shown below.
  If you were on gpt-realtime-1.5, update your model ID to gpt-realtime-2.
""")


# --- Example 1: Text-mode session (no audio hardware required) ---
print("=" * 60)
print("EXAMPLE 1: Text-mode session with gpt-realtime-2")
print("=" * 60)
print()
print("Setting output_modalities=['text'] bypasses audio I/O entirely.")
print("Useful for testing tool use, reasoning effort, and response quality")
print("before wiring up microphone/speaker hardware.")
print()


async def example_1_text_session():
    async with client.realtime.connect(model="gpt-realtime-2") as connection:
        # Configure the session for text-only I/O
        await connection.session.update(session={
            "instructions": (
                "You are a helpful assistant. Be concise — "
                "this is a real-time interaction."
            ),
            "output_modalities": ["text"],
        })

        # Send a text message
        await connection.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is 17 times 23? Show your working."}],
        })
        await connection.response.create()

        # Stream the response events
        print("Response: ", end="", flush=True)
        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                print()
                break

    print()


asyncio.run(example_1_text_session())


# --- Example 2: Audio-mode session pattern ---
print("=" * 60)
print("EXAMPLE 2: Audio-mode session pattern (reference)")
print("=" * 60)
print("""
A full audio loop requires a microphone + speaker harness. Here's the pattern:

```python
import asyncio, base64
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def voice_agent():
    async with client.realtime.connect(model="gpt-realtime-2") as connection:
        # Configure for audio I/O
        await connection.session.update(session={
            "instructions": "You are a helpful voice assistant.",
            "input_modalities": ["audio"],
            "output_modalities": ["audio", "text"],
            "voice": "alloy",
            # Reasoning effort: "minimal" | "low" | "medium" | "high" | "xhigh"
            # Lower effort = lower latency; higher effort = better on complex tasks
            "reasoning_effort": "medium",
            # Preamble: spoken phrase before reasoning begins ("one moment...")
            "preamble": "Sure,",
        })

        # Feed PCM16 audio chunks from your microphone
        # audio_chunk = record_microphone_chunk()  # your harness
        # await connection.input_audio_buffer.append(
        #     audio=base64.b64encode(audio_chunk).decode()
        # )
        # await connection.input_audio_buffer.commit()
        # await connection.response.create()

        # Stream events
        async for event in connection:
            if event.type == "response.output_audio.delta":
                # play_audio_chunk(base64.b64decode(event.delta))  # your harness
                pass
            elif event.type == "response.output_audio_transcript.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                break

asyncio.run(voice_agent())
```

Key points:
  - reasoning_effort controls the latency / quality tradeoff
  - preamble lets the model say a short phrase while it reasons (keeps the call fluent)
  - Parallel tool calls: model can say "checking your calendar now" while calling a tool
  - Always run in an isolated environment; maintain human oversight for high-impact actions
""")


# --- Example 3: Realtime Translate and Whisper patterns ---
print("=" * 60)
print("EXAMPLE 3: gpt-realtime-translate and gpt-realtime-whisper patterns")
print("=" * 60)
print("""
These two models use the same session/event interface as gpt-realtime-2
but are optimized for specific tasks:

gpt-realtime-translate — speech-to-speech translation
─────────────────────────────────────────────────────
async with client.realtime.connect(model="gpt-realtime-translate") as conn:
    await conn.session.update(session={
        "input_language": "es",    # speaker's language (ISO-639-1 or "auto")
        "output_language": "en",   # desired output language
        "input_modalities": ["audio"],
        "output_modalities": ["audio"],
    })
    # same audio feed / event loop as above

Supported input languages:   70+
Supported output languages:  13 (en, es, fr, de, it, pt, nl, pl, ja, ko, zh, ar, ru)
Pricing:                     $0.034/min (flat, no per-token charge)

gpt-realtime-whisper — streaming speech-to-text
────────────────────────────────────────────────
async with client.realtime.connect(model="gpt-realtime-whisper") as conn:
    await conn.session.update(session={
        "input_modalities": ["audio"],
        "output_modalities": ["text"],
    })
    # Events: response.output_text.delta carries the transcript as it's spoken

Compared to batch Whisper: lower latency, partial transcripts while speaking.
Compared to input_audio_transcription in a gpt-realtime-2 session: gpt-realtime-
whisper is a dedicated transcription-only model at half the price ($0.017/min).
Pricing: $0.017/min
""")

print("=" * 60)
print("KEY CONCEPTS")
print("=" * 60)
print("""
Session setup:
  client.realtime.connect(model="<model-id>")  — WebSocket session
  connection.session.update(session={...})      — configure instructions, modalities, voice
  connection.conversation.item.create(item={...}) — add a turn
  connection.response.create()                  — trigger model response

GA vs Beta interface:
  Beta (removed May 12, 2026): required OpenAI-Beta: realtime=v1 header
  GA (current):                no special header; use session.type events, new event names

Event names (GA):
  response.output_text.delta            — text token streamed
  response.output_audio.delta           — audio chunk (base64)
  response.output_audio_transcript.delta — transcript of audio
  response.done                         — model finished response

Choosing a model:
  gpt-realtime-2           — any voice agent that needs to reason or call tools
  gpt-realtime-translate   — live multilingual translation (flat $/min)
  gpt-realtime-whisper     — live transcription only (cheapest option)

Pricing note:
  gpt-realtime-2 bills per audio token (like text tokens but for speech).
  Cached audio input is $0.40/M tokens — structure calls to benefit from caching.
  gpt-realtime-translate and gpt-realtime-whisper bill per minute (simpler).
""")

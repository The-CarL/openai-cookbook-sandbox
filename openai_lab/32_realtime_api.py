"""Exercise 32: Realtime API 2 — gpt-realtime-2, translate, and whisper (GA May 7, 2026).

Three new models replaced the Realtime API Beta (gpt-realtime-1.5) when it was
removed on May 7, 2026:
  gpt-realtime-2          — Voice agent with GPT-5-class reasoning, 128K ctx
  gpt-realtime-translate  — Live speech translation (70+ → 13 languages)
  gpt-realtime-whisper    — Streaming speech-to-text transcription

Connection: client.realtime.connect()  (old beta used client.beta.realtime.connect())
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()

PRICING = {
    "gpt-realtime-2":         {"audio_input": 32.00, "cached_input": 0.40, "audio_output": 64.00},
    "gpt-realtime-translate": {"per_minute": 0.034},
    "gpt-realtime-whisper":   {"per_minute": 0.017},
}

# --- Example 1: Text conversation with gpt-realtime-2 ---
print("=" * 60)
print("EXAMPLE 1: Text conversation with gpt-realtime-2")
print("=" * 60)
print()
print("modalities=['text'] exercises the model without audio hardware.")
print()


async def example_text():
    async with client.realtime.connect(model="gpt-realtime-2") as connection:
        await connection.session.update(session={
            "modalities": ["text"],
            "instructions": "You are a concise assistant. Reply in one sentence.",
            "temperature": 0.8,
        })

        await connection.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is new in gpt-realtime-2 vs the old Realtime API?"}],
        })
        await connection.response.create()

        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.output_text.done":
                print()
            elif event.type == "response.done":
                break


asyncio.run(example_text())

# --- Example 2: Voice session reference pattern ---
print()
print("=" * 60)
print("EXAMPLE 2: Voice session (reference pattern — requires audio harness)")
print("=" * 60)
print("""
```python
async def voice_agent(instructions: str):
    async with client.realtime.connect(model="gpt-realtime-2") as connection:
        await connection.session.update(session={
            "modalities": ["text", "audio"],
            "instructions": instructions,
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",      # Server-side voice activity detection
                "threshold": 0.5,
                "silence_duration_ms": 800,
            },
        })

        async for event in connection:
            if event.type == "input_audio_buffer.speech_started":
                print("[User speaking...]")
            elif event.type == "response.audio.delta":
                play_audio_chunk(event.delta)  # Your harness function
            elif event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)  # Optional transcript
            elif event.type == "response.done":
                pass  # Continue loop for next utterance

# New in gpt-realtime-2:
#   reasoning_effort  "low" | "medium" | "high" — how hard the model thinks
#   preambles         Short phrases while reasoning ("let me check that...")
#   parallel tools    Multiple tool calls in one turn
#   128K context      4x longer sessions vs gpt-realtime-1.5 (32K)
```
""")

# --- Example 3: gpt-realtime-translate ---
print("=" * 60)
print("EXAMPLE 3: gpt-realtime-translate (reference pattern)")
print("=" * 60)
print("""
```python
async def live_translation(output_language: str = "English"):
    async with client.realtime.connect(model="gpt-realtime-translate") as connection:
        await connection.session.update(session={
            "modalities": ["text", "audio"],
            "input_language": "auto",        # Detect automatically from 70+ languages
            "output_language": output_language,  # One of 13 supported output languages
        })

        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)  # Translated text
            elif event.type == "response.audio.delta":
                play_audio_chunk(event.delta)            # Translated speech

# Pricing: $0.034 / minute of audio
```
""")

# --- Example 4: gpt-realtime-whisper ---
print("=" * 60)
print("EXAMPLE 4: gpt-realtime-whisper — streaming STT (reference pattern)")
print("=" * 60)
print("""
```python
async def streaming_transcription():
    async with client.realtime.connect(model="gpt-realtime-whisper") as connection:
        await connection.session.update(session={
            "language": "auto",      # Auto-detect spoken language
            "modalities": ["text"],  # Transcription only — no audio output
        })

        async for event in connection:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)  # Live transcript
            elif event.type == "response.output_text.done":
                print()  # End of utterance

# Pricing: $0.017 / minute of audio
# Use cases: live captions, meeting notes, agent listening layer
```
""")

# --- Key concepts ---
print("=" * 60)
print("REALTIME API 2 KEY CONCEPTS")
print("=" * 60)
print(f"""
Models and pricing (GA May 7, 2026):
  gpt-realtime-2
    Audio input:   ${PRICING['gpt-realtime-2']['audio_input']:.2f} / M tokens
    Cached input:  ${PRICING['gpt-realtime-2']['cached_input']:.2f} / M tokens
    Audio output:  ${PRICING['gpt-realtime-2']['audio_output']:.2f} / M tokens
  gpt-realtime-translate  ${PRICING['gpt-realtime-translate']['per_minute']:.4f} / min
  gpt-realtime-whisper    ${PRICING['gpt-realtime-whisper']['per_minute']:.4f} / min

Connection and event loop:
  async with client.realtime.connect(model=<model>) as connection:
    Old beta (removed): client.beta.realtime.connect()

Key events:
  session.created                   Connection ready
  input_audio_buffer.speech_started VAD detected speech
  response.output_text.delta        Text token streaming
  response.audio.delta              Audio chunk (PCM16) streaming
  response.done                     Turn complete

gpt-realtime-2 vs gpt-realtime-1.5:
  Context:      128K tokens (was 32K)
  Reasoning:    GPT-5-class with configurable reasoning_effort
  Preambles:    Model speaks "let me check that..." while thinking
  Tools:        Parallel calls + verbal transparency ("looking that up...")
""")

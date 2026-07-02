"""Exercise 32: Realtime API GA — gpt-realtime-2, gpt-realtime-translate, gpt-realtime-whisper.

May 7, 2026: Realtime API graduated to general availability with three new models:

  gpt-realtime-2         — GPT-5-class reasoning voice model. Accepts text/audio/image;
                           outputs text and audio. First voice model that can reason before
                           it speaks. Priced per token (audio: $32/$64 per M, text: $4/$24 per M).

  gpt-realtime-translate — Live speech-to-speech translation across 70+ input and 13 output
                           languages. Keeps pace with the speaker in real time. $0.034/min.

  gpt-realtime-whisper   — Streaming speech-to-text (transcription only, no generation).
                           Transcribes live as the speaker talks. $0.017/min.

The old Realtime API Beta (gpt-realtime-1.5) was deprecated May 12, 2026.
The GA interface adds configurable reasoning effort, improved reliability, and
cleaner developer ergonomics — session config is now a proper typed object.

Reference: https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/
"""

import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()


# --- Overview ---
print("=" * 60)
print("REALTIME API GA — THREE NEW MODELS (May 7, 2026)")
print("=" * 60)
print("""
Model lineup:
  gpt-realtime-2         Voice + reasoning. The flagship voice agent model.
  gpt-realtime-translate Live speech translation (70+ → 13 languages).
  gpt-realtime-whisper   Streaming transcription only (no audio output).

All three replaced the deprecated beta (gpt-realtime-1.5).

The Realtime API uses WebSocket connections — each session is a
persistent bidirectional stream, not a request/response pair.
""")


# --- Example 1: Text-mode call with gpt-realtime-2 ---
# gpt-realtime-2 supports text-only mode — useful for testing without
# audio hardware. Set modalities=["text"] in session config.

async def example_text_mode():
    print("=" * 60)
    print("EXAMPLE 1: gpt-realtime-2 in text mode (no audio required)")
    print("=" * 60)
    print()

    async with client.beta.realtime.connect(model="gpt-realtime-2") as conn:
        # Configure session: text only, medium reasoning effort
        await conn.session.update(session={
            "modalities": ["text"],
            "instructions": "You are a concise assistant. One sentence only.",
            "reasoning": {"effort": "medium"},
        })

        # Add a user message
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the Realtime API best suited for?"}],
        })

        # Trigger a response
        await conn.response.create()

        # Consume events until response is complete
        final_text = []
        async for event in conn:
            if event.type == "response.text.delta":
                final_text.append(event.delta)
            elif event.type == "response.done":
                break

        print(f"gpt-realtime-2 says: {''.join(final_text)}")
        print()

        # Show usage from the response.done event (usage is in event.response)
        print("Key points from session:")
        print("  - WebSocket kept open for the full turn")
        print("  - reasoning.effort=medium: model thinks before answering")
        print("  - In audio mode, set modalities=['text','audio'] and")
        print("    stream audio chunks via input_audio_buffer.append events")


# --- Example 2: Audio streaming pattern (reference) ---
# Requires a microphone / audio pipeline. Shown as a code block.

AUDIO_PATTERN = '''
"""Full audio streaming pattern with gpt-realtime-2."""

import asyncio, base64
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def voice_session():
    async with client.beta.realtime.connect(model="gpt-realtime-2") as conn:
        await conn.session.update(session={
            "modalities": ["text", "audio"],
            "voice": "shimmer",          # audio output voice
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",   # server-side voice activity detection
                "threshold": 0.5,
                "silence_duration_ms": 800,
            },
            "reasoning": {"effort": "medium"},
        })

        # Stream microphone audio to the server in chunks
        async def send_audio(mic_stream):
            async for chunk in mic_stream:
                await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())

        # Receive events and route them
        async def receive_events(speaker):
            async for event in conn:
                if event.type == "response.audio.delta":
                    # Play this audio chunk
                    speaker.write(base64.b64decode(event.delta))
                elif event.type == "response.text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.done":
                    print()  # newline after transcript

        mic = get_microphone_stream()   # your harness
        speaker = get_speaker()         # your harness
        await asyncio.gather(send_audio(mic), receive_events(speaker))

asyncio.run(voice_session())
'''

# --- Example 3: gpt-realtime-translate pattern ---

TRANSLATE_PATTERN = '''
"""Live speech translation with gpt-realtime-translate."""

async def translate_session():
    async with client.beta.realtime.connect(model="gpt-realtime-translate") as conn:
        await conn.session.update(session={
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "translation": {
                "source_language": "auto",   # detect input language
                "target_language": "en",     # one of 13 supported output langs
            },
            # No turn_detection needed — translate mode streams continuously
        })

        async def send_audio(mic_stream):
            async for chunk in mic_stream:
                await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())

        async def receive_events(speaker):
            async for event in conn:
                if event.type == "response.audio.delta":
                    speaker.write(base64.b64decode(event.delta))

        await asyncio.gather(send_audio(mic), receive_events(speaker))
'''

# --- Example 4: gpt-realtime-whisper pattern ---

WHISPER_PATTERN = '''
"""Streaming transcription with gpt-realtime-whisper (text output only)."""

async def transcribe_session():
    async with client.beta.realtime.connect(model="gpt-realtime-whisper") as conn:
        await conn.session.update(session={
            "modalities": ["text"],           # text output only
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"language": "en"},
        })

        async def send_audio(mic_stream):
            async for chunk in mic_stream:
                await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())

        async def receive_events():
            async for event in conn:
                if event.type == "conversation.item.input_audio_transcription.delta":
                    # Streaming partial transcript as speaker talks
                    print(event.delta, end="", flush=True)
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    print()  # newline after completed utterance

        await asyncio.gather(send_audio(mic), receive_events())
'''


async def main():
    await example_text_mode()

    print("=" * 60)
    print("EXAMPLE 2: Full audio streaming pattern (gpt-realtime-2)")
    print("=" * 60)
    print("Requires microphone + speaker. Reference pattern:")
    print(AUDIO_PATTERN)

    print("=" * 60)
    print("EXAMPLE 3: Live translation (gpt-realtime-translate)")
    print("=" * 60)
    print(TRANSLATE_PATTERN)

    print("=" * 60)
    print("EXAMPLE 4: Streaming transcription (gpt-realtime-whisper)")
    print("=" * 60)
    print(WHISPER_PATTERN)

    print("=" * 60)
    print("REALTIME API KEY CONCEPTS")
    print("=" * 60)
    print("""
Session lifecycle:
  connect()       — Opens WebSocket. One session per call.
  session.update  — Configures modalities, voice, VAD, reasoning effort.
  item.create     — Adds a message to the conversation.
  response.create — Triggers the model to respond.
  Iterate conn    — Yields typed events (text delta, audio delta, done).

Event types (gpt-realtime-2):
  response.text.delta          — Text token streamed
  response.audio.delta         — Audio chunk (base64 PCM)
  response.done                — Turn complete; usage info attached
  input_speech_started         — VAD detected speech start
  input_speech_stopped         — VAD detected speech end
  conversation.item.created    — A new item was appended

Pricing (per 1M tokens unless noted):
  gpt-realtime-2      audio in $32 / audio out $64, text in $4 / text out $24
  gpt-realtime-2      cached audio in: $0.40
  gpt-realtime-translate       $0.034/min
  gpt-realtime-whisper         $0.017/min

Model capability comparison:
  gpt-realtime-2         Full reasoning. Use for voice agents that handle
                         complex questions, multi-turn tasks, or tool use.
  gpt-realtime-translate Dedicated translation. Lower latency than using
                         gpt-realtime-2 with a translate instruction.
  gpt-realtime-whisper   Transcription only (text output). Cheapest option
                         when downstream processing handles NLU.

Beta vs GA migration notes:
  - Replace model="gpt-realtime-1.5" with one of the three GA models above.
  - Session config is now a typed object (conn.session.update(session={...})).
  - input_audio_buffer.append() replaces the raw "input_audio_buffer.append"
    event message format from the beta.
  - See: https://platform.openai.com/docs/guides/realtime-beta-migration
""")


if __name__ == "__main__":
    asyncio.run(main())

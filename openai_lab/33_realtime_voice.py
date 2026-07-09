"""Exercise 33: Realtime API — gpt-realtime-2.1, gpt-realtime-2.1-mini, translate, whisper.

The Realtime API (GA May 7, 2026) delivers low-latency, bidirectional voice
agents via a persistent WebSocket session. Four models serve distinct jobs:

  gpt-realtime-2.1       — GPT-5-class reasoning for full voice agents (Jul 6, 2026).
                           Improves on 2.0: better alphanumeric recognition, noise
                           handling, interruption behavior; ≥25% lower p95 latency.
                           Billing: audio tokens (input ~$32/M, output ~$128/M).
  gpt-realtime-2.1-mini  — Reasoning + tool use at ~6x lower cost (Jul 6, 2026).
                           New: 2.0-mini lacked reasoning; 2.1-mini closes that gap.
                           Billing: audio tokens (cheaper tier than 2.1).
  gpt-realtime-translate — Live speech-to-speech translation, 70+ → 13 languages.
                           Billing: per minute of input audio.
  gpt-realtime-whisper   — Streaming speech-to-text transcription.
                           Billing: per minute of input audio.

Unlike the Responses API (REST + stateless), the Realtime API is WebSocket-based:
  - Persistent session — no previous_response_id chaining needed.
  - Bidirectional: you stream audio in while transcript/audio events stream out.
  - gpt-realtime-2.1 supports text-only mode — useful for testing without hardware.

Use gpt-realtime-2.1 (not 2.0) for new production deployments.
Use gpt-realtime-2.1-mini when cost matters and reasoning/tool-use are still needed.

Requires: uv add websockets
Reference: https://developers.openai.com/api/docs/realtime
"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]

# ---- helpers ----------------------------------------------------------------

def _fmt_event(e: dict) -> str:
    etype = e.get("type", "?")
    if etype == "session.created":
        sid = e.get("session", {}).get("id", "?")
        return f"  session.created  id={sid}"
    if etype == "response.text.delta":
        return f"  text.delta       '{e.get('delta', '')}'"
    if etype == "response.text.done":
        return f"  text.done"
    if etype == "response.done":
        usage = e.get("response", {}).get("usage", {})
        return f"  response.done    usage={usage}"
    if etype == "error":
        return f"  ERROR            {e.get('error', {})}"
    return f"  {etype}"


# ---- Example 1: Text conversation with gpt-realtime-2 ----------------------

async def example_1_text_convo():
    """Connect to gpt-realtime-2.1 in text-only mode (no audio hardware needed)."""
    try:
        import websockets
    except ImportError:
        print("  [skip] 'websockets' not installed — run: uv add websockets")
        return

    print("=" * 60)
    print("EXAMPLE 1: Text conversation with gpt-realtime-2.1")
    print("=" * 60)
    print()
    print("Connecting to wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1")
    print()

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with websockets.connect(url, additional_headers=headers) as ws:
        # Step 1: wait for session.created
        evt = json.loads(await ws.recv())
        print(_fmt_event(evt))

        # Step 2: configure session for text-only (no audio encoding/decoding)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": (
                    "You are a concise AI assistant. Keep all answers under "
                    "three sentences."
                ),
            },
        }))
        evt = json.loads(await ws.recv())
        print(_fmt_event(evt))  # session.updated

        # Step 3: add a user message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is the OpenAI Realtime API and when did it go GA?"}
                ],
            },
        }))
        evt = json.loads(await ws.recv())
        print(_fmt_event(evt))  # conversation.item.created

        # Step 4: trigger model response
        await ws.send(json.dumps({"type": "response.create"}))

        # Step 5: stream the response
        full_text = ""
        print("\n  [response streaming]")
        async for raw in ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")
            print(_fmt_event(evt))
            if etype == "response.text.delta":
                full_text += evt.get("delta", "")
            elif etype == "response.done":
                break

        print(f"\nFull response:\n  {full_text}")
    print()


# ---- Example 2: gpt-realtime-translate pattern (audio required) -------------

def example_2_translate_pattern():
    print("=" * 60)
    print("EXAMPLE 2: gpt-realtime-translate (speech-to-speech, 70+ languages)")
    print("=" * 60)
    print("""
gpt-realtime-translate is a specialized model for live speech translation.
It accepts audio input and returns translated audio — no intermediate text.

Connection and session setup:

  url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-translate"

  # Configure session (source lang optional — model auto-detects if omitted)
  await ws.send(json.dumps({
      "type": "session.update",
      "session": {
          "input_audio_format": "pcm16",          # 16-bit PCM, 24 kHz mono
          "output_audio_format": "pcm16",
          "input_language": "fr",                  # or omit for auto-detect
          "output_language": "en",                 # target language
      },
  }))

  # Stream audio in chunks
  import base64
  for chunk in mic_stream():        # your audio capture loop
      await ws.send(json.dumps({
          "type": "input_audio_buffer.append",
          "audio": base64.b64encode(chunk).decode(),
      }))

  await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
  await ws.send(json.dumps({"type": "response.create"}))

  # Receive translated audio
  async for raw in ws:
      evt = json.loads(raw)
      if evt["type"] == "response.audio.delta":
          play_audio(base64.b64decode(evt["delta"]))
      elif evt["type"] == "response.done":
          break

Billing: per minute of input audio (not per token).
Supported: 70+ input languages → 13 output languages (EN, ES, FR, DE, IT,
           JA, KO, NL, PL, PT, RU, TR, ZH).
""")


# ---- Example 3: gpt-realtime-whisper pattern (streaming transcription) ------

def example_3_whisper_pattern():
    print("=" * 60)
    print("EXAMPLE 3: gpt-realtime-whisper (streaming speech-to-text)")
    print("=" * 60)
    print("""
gpt-realtime-whisper streams transcriptions live as the speaker talks —
much lower latency than batch Whisper, at the cost of higher per-minute price.

Connection and session setup:

  url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-whisper"

  await ws.send(json.dumps({
      "type": "session.update",
      "session": {
          "input_audio_format": "pcm16",
          "language": "en",            # or omit for auto-detect
          "turn_detection": {
              "type": "server_vad",    # automatic utterance segmentation
          },
      },
  }))

  for chunk in mic_stream():
      await ws.send(json.dumps({
          "type": "input_audio_buffer.append",
          "audio": base64.b64encode(chunk).decode(),
      }))

  # Receive transcript incrementally
  async for raw in ws:
      evt = json.loads(raw)
      if evt["type"] == "conversation.item.input_audio_transcription.delta":
          print(evt["delta"], end="", flush=True)
      elif evt["type"] == "input_audio_buffer.speech_stopped":
          print()   # end of utterance

Billing: per minute of input audio.
Voice Activity Detection (server_vad) automatically segments utterances
so you don't have to commit the buffer manually.
""")


# ---- Model comparison and choosing -----------------------------------------

def summary():
    print("=" * 60)
    print("REALTIME MODEL COMPARISON (updated Jul 6, 2026)")
    print("=" * 60)
    print("""
Model                    Use case                            Billing
──────────────────────────────────────────────────────────────────────
gpt-realtime-2.1         Full voice agent with GPT-5          Audio tokens
  (Jul 6, 2026)          reasoning, function calls,           (in ~$32/M,
  Recommended over 2.0   interruption handling.               out ~$128/M)
                         Better alphanumeric, noise handling,
                         interruptions; ≥25% lower p95 latency.

gpt-realtime-2.1-mini    Reasoning + tool use at lower cost.  Audio tokens
  (Jul 6, 2026)          2.0-mini lacked reasoning;           (~6x cheaper
  New: reasoning at      2.1-mini closes that gap.            than 2.1)
  mini price             Same ≥25% latency improvement.

gpt-realtime-translate   Live speech-to-speech translation    Per minute
                         70+ input → 13 output languages      of input

gpt-realtime-whisper     Streaming speech-to-text             Per minute
                         Lowest-latency transcription         of input

Key differences from the Responses API:
  ✗ Not REST — a persistent WebSocket session per conversation
  ✗ No previous_response_id — the session IS the context
  ✓ Bidirectional: send audio in, receive audio+transcript simultaneously
  ✓ Server VAD: model detects end-of-speech automatically
  ✓ Function calling works in gpt-realtime-2.1 / 2.1-mini

When to use which:
  gpt-realtime-2.1       — Customer support bots, voice assistants, any
                           agent that needs to reason, use tools, or handle
                           interruptions in real time. Default for new builds.
  gpt-realtime-2.1-mini  — Same jobs as 2.1 when per-minute cost matters.
                           Now supports reasoning + tool use (unlike 2.0-mini).
  gpt-realtime-translate — Call center translation, live interpreting,
                           multilingual customer service.
  gpt-realtime-whisper   — Meeting transcription, live captioning,
                           voice-to-text where you control the LLM layer.

Prompt caching:
  gpt-realtime-2.1 supports prompt caching on audio tokens.
  Typical discount: ~98% on cached input audio.

Session timeout:
  Sessions expire after ~30 minutes of inactivity. Reconnect and replay
  the session config (session.update) — conversation history is not
  automatically restored.

Model aliases:
  gpt-realtime-2 still works and routes to the 2.0 generation. For new
  production deployments, pin to gpt-realtime-2.1 explicitly.
""")


# ---- Entry point ------------------------------------------------------------

async def main():
    await example_1_text_convo()
    example_2_translate_pattern()
    example_3_whisper_pattern()
    summary()


if __name__ == "__main__":
    asyncio.run(main())

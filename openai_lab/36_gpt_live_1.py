"""Exercise 36: GPT-Live-1 — full-duplex voice model (Sept 10 2026).

GPT-Live-1 replaces the conventional speech pipeline (STT → LLM → TTS) with a
single integrated model that listens and speaks at the same time. Callers can
interrupt without breaking the conversation.

Key differences from gpt-realtime-2 (Exercise 33):

  gpt-realtime-2     — Audio token billing (~$32/M input, ~$128/M output).
                        Half-duplex: caller interrupts by pausing the stream.
                        Baked-in reasoning — one model does everything.

  GPT-Live-1         — Flat per-minute billing: $0.05/min for the voice layer.
                        Full-duplex: simultaneous listen + speak; caller can
                        interrupt mid-sentence without a pipeline reset.
                        Modular: delegates deep reasoning and tool calls to a
                        backend agent (Agents API, Codex, Responses API).
                        Scores 30 pp higher than gpt-realtime-2.1 on Full Duplex Bench.

Connection options: WebSocket, WebRTC (browser), SIP/telephony.

Requires: pip install "openai[realtime]>=3.8.0"
Reference: https://openai.com/index/introducing-gpt-live-1-in-the-api/
"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-live-1"


# ---- helpers ----------------------------------------------------------------

def _fmt(e: dict) -> str:
    etype = e.get("type", "?")
    if etype == "session.created":
        sid = e.get("session", {}).get("id", "?")
        return f"  session.created  id={sid}"
    if etype == "response.audio_transcript.delta":
        return f"  transcript.delta '{e.get('delta', '')}'"
    if etype == "response.audio_transcript.done":
        transcript = e.get("transcript", "")
        return f"  transcript.done  '{transcript[:80]}'"
    if etype == "response.done":
        usage = e.get("response", {}).get("usage", {})
        dur = e.get("response", {}).get("duration_minutes", None)
        return f"  response.done    usage={usage} duration_min={dur}"
    if etype == "error":
        return f"  ERROR            {e.get('error', {})}"
    return f"  {etype}"


# ---- Example 1: Connection setup and session config -------------------------

async def example_1_session_config():
    """Show GPT-Live-1 session configuration — text-only test (no mic needed)."""
    try:
        import websockets
    except ImportError:
        print("  [skip] 'websockets' not installed — run: uv add websockets")
        return

    print("Connecting to gpt-live-1 via WebSocket...")
    print("Using text-only turn for testability (no audio hardware needed).")
    print()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        # Configure the session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "input_audio_transcription": {"model": "whisper-1"},
            },
        }))

        # Send a text-mode input (simulates conversation turn without a mic)
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is GPT-Live-1 and how does it differ from gpt-realtime-2?"}],
            },
        }))
        await ws.send(json.dumps({"type": "response.create"}))

        # Collect events until response.done
        transcript_parts = []
        async for raw in ws:
            e = json.loads(raw)
            print(_fmt(e))
            if e.get("type") == "response.audio_transcript.delta":
                transcript_parts.append(e.get("delta", ""))
            if e.get("type") == "response.done":
                break

    full_transcript = "".join(transcript_parts)
    if full_transcript:
        print(f"\nFull transcript:\n  {full_transcript}")


# ---- Example 2: Architecture comparison -------------------------------------

def example_2_architecture():
    """Print a side-by-side architecture comparison (no API call needed)."""
    print("=" * 60)
    print("VOICE PIPELINE COMPARISON")
    print("=" * 60)
    print("""
  Traditional 3-stage pipeline:
    Caller audio → [STT] → text → [LLM] → text → [TTS] → audio
    Latency: ~800-1200 ms per turn. Interruption breaks all three stages.

  gpt-realtime-2 (Exercise 33):
    Caller audio → [gpt-realtime-2] → audio/text
    Latency: ~300-500 ms. Half-duplex: interrupt by stopping your stream.
    Token billing: ~$32/M input audio, ~$128/M output audio tokens.

  GPT-Live-1 (this exercise):
    Caller audio → [GPT-Live-1] → audio/text     (voice layer: $0.05/min)
                         ↓
                 [backend agent]                  (token billing at standard rates)
                         ↑
    Simultaneous send/receive — caller can speak while model is still speaking.
    Interruptions are first-class. Score: +30 pp vs gpt-realtime-2.1 on Full Duplex Bench.

  When to choose:
    gpt-realtime-2  — Low per-minute cost at short turn lengths; all-in-one reasoning.
    GPT-Live-1      — Natural conversational feel; delegation to a powerful backend agent;
                      high-interrupt scenarios (e.g., call centres, voice assistants).

  Pricing example (10-minute conversation):
    gpt-realtime-2:  rough token cost depends on audio length (varies)
    GPT-Live-1:      $0.05/min × 10 min = $0.50 (voice layer)
                     + backend agent tokens at standard Responses API rates
""")


# ---- Example 3: Key session parameters (reference) -------------------------

def example_3_params():
    """Print the key GPT-Live-1 session.update parameters."""
    print("=" * 60)
    print("GPT-LIVE-1 SESSION PARAMETERS")
    print("=" * 60)
    reference = {
        "modalities": ["text", "audio"],
        "voice": "alloy",          # alloy / echo / shimmer / fable + 12 new voices
        "turn_detection": {
            "type": "server_vad",  # server-side voice activity detection
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        },
        "input_audio_format": "pcm16",      # pcm16 | g711_ulaw | g711_alaw
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "instructions": "You are a helpful voice assistant.",
        "backend_agent": {                  # Optional: delegate reasoning to an agent
            "type": "responses_api",        # responses_api | agents_api
            "model": "gpt-6-astra",
        },
    }
    print(json.dumps(reference, indent=2))
    print("""
Notes:
  - backend_agent.type="agents_api" connects to client.beta.agents (Exercise 35)
  - backend_agent.type="responses_api" delegates reasoning to a Responses API model
  - Omit backend_agent to use GPT-Live-1's built-in reasoning (no separate backend)
  - Connection: WebSocket, WebRTC, or SIP — same session.update payload
  - Billing: $0.05/min for the voice layer only; backend agent billed separately
""")


# ---- Run all examples -------------------------------------------------------

async def main():
    print("=" * 60)
    print("EXAMPLE 1: Session connection and text-mode turn")
    print("=" * 60)
    print()
    await example_1_session_config()

    print()
    example_2_architecture()
    print()
    example_3_params()


if __name__ == "__main__":
    asyncio.run(main())

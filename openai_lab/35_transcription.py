"""Exercise 35: GPT-Transcribe and GPT-Live-Transcribe (July 28, 2026).

Two new transcription models replace gpt-4o-transcribe for new builds:

  gpt-transcribe      — async file transcription, REST API, $0.0045/min
  gpt-live-transcribe — streaming live transcription, WebSocket, $0.017/min

Key improvements over gpt-4o-transcribe:
  - Lower WER (3.31% vs ~5% for gpt-4o-transcribe on real-world audio)
  - Three context types: prompt (free-form), keywords, languages
  - Handles heavy accents, background noise, and specialized terminology better
  - gpt-live-transcribe supports tunable latency and language hints per turn

Pricing comparison (per minute of audio):
  gpt-4o-transcribe     $0.006   (previous default)
  gpt-transcribe        $0.0045  (25% cheaper, higher accuracy)
  gpt-live-transcribe   $0.017   (streaming; same as gpt-realtime-whisper)
  gpt-realtime-whisper  $0.017   (covered in exercise 33)

This exercise covers gpt-transcribe via the REST Audio API.
For gpt-live-transcribe, see the WebSocket pattern in exercise 33.
"""

import io
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# ── helpers ──────────────────────────────────────────────────────────────────

def synthesize_audio(text: str) -> bytes:
    """Generate test audio via TTS so the exercise is self-contained."""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="mp3",
    )
    return response.content


# ── Example 1: Basic transcription ───────────────────────────────────────────

print("=" * 60)
print("EXAMPLE 1: Basic file transcription with gpt-transcribe")
print("=" * 60)
print()

sample_text = (
    "Welcome to the OpenAI API overview. Today we'll cover the "
    "Responses API, the Agents SDK, and the new GPT-5.6 model family "
    "including Sol, Terra, and Luna. We'll also look at prompt caching "
    "and how it affects your total cost of ownership."
)

print("Generating test audio via TTS...")
audio_bytes = synthesize_audio(sample_text)
print(f"Audio size: {len(audio_bytes):,} bytes")
print()

audio_file = io.BytesIO(audio_bytes)
audio_file.name = "test.mp3"

transcript = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=audio_file,
)

print(f"Transcript:\n  {transcript.text}")
print()


# ── Example 2: Context hints (prompt + keywords + languages) ─────────────────

print("=" * 60)
print("EXAMPLE 2: Context hints — prompt, keywords, languages")
print("=" * 60)
print()
print("Context types accepted by gpt-transcribe:")
print("  prompt    — free-form description of the recording (topic, setting)")
print("  keywords  — literal terms that may appear (product names, acronyms)")
print("  languages — list of expected input languages; replaces singular 'language'")
print()

tech_audio_text = (
    "The p99 latency for our RAG pipeline jumped from 200ms to 600ms after "
    "switching to text-embedding-3-large. We're investigating chunking strategy "
    "and HNSW index tuning. The MTTR for this incident was under two hours "
    "thanks to our on-call rotation."
)

audio_bytes2 = synthesize_audio(tech_audio_text)
audio_file2 = io.BytesIO(audio_bytes2)
audio_file2.name = "tech.mp3"

transcript2 = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=audio_file2,
    prompt="Engineering postmortem call discussing a RAG pipeline latency incident.",
    keywords=["RAG", "p99", "HNSW", "text-embedding-3-large", "MTTR"],
    languages=["en"],
)

print(f"Transcript with hints:\n  {transcript2.text}")
print()


# ── Example 3: Compare gpt-transcribe vs gpt-4o-transcribe ───────────────────

print("=" * 60)
print("EXAMPLE 3: gpt-transcribe vs gpt-4o-transcribe")
print("=" * 60)
print()

audio_bytes3 = synthesize_audio(sample_text)

audio_file3a = io.BytesIO(audio_bytes3)
audio_file3a.name = "compare.mp3"
t_new = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=audio_file3a,
)

audio_file3b = io.BytesIO(audio_bytes3)
audio_file3b.name = "compare.mp3"
t_old = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file3b,
)

print(f"gpt-transcribe:     {t_new.text[:120]}")
print(f"gpt-4o-transcribe:  {t_old.text[:120]}")
print()

audio_minutes = len(audio_bytes3) / (128 * 1000 / 8 * 60)
cost_new = audio_minutes * 0.0045
cost_old = audio_minutes * 0.006
print(f"Estimated audio duration: ~{audio_minutes:.3f} min")
print(f"  gpt-transcribe cost:    ${cost_new:.6f}  ($0.0045/min)")
print(f"  gpt-4o-transcribe cost: ${cost_old:.6f}  ($0.006/min)")
print(f"  Savings:                {(1 - cost_new/cost_old):.0%}")
print()


# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("TRANSCRIPTION MODEL SUMMARY")
print("=" * 60)
print("""
gpt-transcribe (July 28, 2026)
  API:       client.audio.transcriptions.create(model="gpt-transcribe", ...)
  Use for:   Completed audio files, batch workloads, async pipelines
  Pricing:   $0.0045/min (25% cheaper than gpt-4o-transcribe)
  Context:   prompt= (free-form), keywords= (list), languages= (list)
  Note:      'languages' is a list; don't also send 'language' (singular)

gpt-live-transcribe (July 28, 2026)
  API:       WebSocket / Realtime API (same pattern as exercise 33)
  Use for:   Live captions, real-time voice apps, call center streaming
  Pricing:   $0.017/min (same as gpt-realtime-whisper)
  Features:  Tunable latency, keyword hints, multi-language hints per turn
  Reference: Exercise 33 for the WebSocket session + event loop pattern

Model selection guide:
  New async transcription work  → gpt-transcribe
  Live / streaming transcription → gpt-live-transcribe
  Full voice agent (LLM + audio) → gpt-realtime-2 (exercise 33)
  Legacy workloads              → gpt-4o-transcribe still supported
""")

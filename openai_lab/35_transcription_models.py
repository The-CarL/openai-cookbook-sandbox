"""Exercise 35: gpt-transcribe and gpt-live-transcribe (July 28, 2026).

Two new speech-to-text models replace the older whisper/4o-transcribe path:

  gpt-transcribe       — Async file transcription and Realtime committed turns.
                         ~halves whisper-1 WER on CommonVoice; $0.0045/min.
  gpt-live-transcribe  — Low-latency streaming transcription over WebSocket.
                         $0.017/min; mirrors the gpt-realtime-whisper API shape.

Both support:
  - Free-form transcription context (domain vocabulary, speaker names, etc.)
  - Keyword hints to anchor recognition of specialized terms
  - Multiple expected_input_languages for multilingual audio
  - Speaker diarization via response_format="diarized_json"

gpt-transcribe uses the same REST endpoint as whisper-1 and gpt-4o-transcribe,
so swapping the model string is the only change required.

gpt-live-transcribe is WebSocket-based; the pattern mirrors exercise 33 (Realtime
API v2) — a persistent session with streamed audio in and transcript events out.
This exercise shows the file-transcription path only; see ex. 33 for the
WebSocket loop pattern.
"""

import os
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# ---- helpers ----------------------------------------------------------------

def _generate_test_audio(text: str, path: str) -> None:
    """Write a short TTS clip to path using the default TTS voice."""
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
    ) as resp:
        resp.stream_to_file(path)


def _transcribe(model: str, path: str, **kwargs) -> dict:
    """Return a dict with model name, text, and (if available) WER-relevant fields."""
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(model=model, file=f, **kwargs)
    return {"model": model, "text": result.text}


# ---- Example 1: Basic file transcription ------------------------------------

print("=" * 60)
print("EXAMPLE 1: Basic file transcription — gpt-transcribe vs whisper-1")
print("=" * 60)
print()

SAMPLE_TEXT = (
    "Acme Corp's Q3 p99 latency is 612 milliseconds, up from 198 milliseconds "
    "in Q2. The team suspects a regression in the gpt-5.6-terra embedding pipeline. "
    "The on-call engineer, Priya Raghunathan, will investigate the FAISS index "
    "rebuild scheduled for 02:00 UTC."
)

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
    audio_path = tf.name

try:
    print("Generating test audio via TTS…")
    _generate_test_audio(SAMPLE_TEXT, audio_path)

    print(f"\nReference text:\n  {SAMPLE_TEXT}\n")

    for model in ("whisper-1", "gpt-transcribe"):
        result = _transcribe(model, audio_path)
        print(f"[{model}]\n  {result['text']}\n")

finally:
    os.unlink(audio_path)


# ---- Example 2: Context and keyword hints -----------------------------------

print("=" * 60)
print("EXAMPLE 2: Context + keyword hints")
print("=" * 60)
print()
print(
    "Pass `prompt` for free-form domain context (speaker names, product terms).\n"
    "Pass `keywords` as a list of strings the model should recognise exactly.\n"
    "This is the primary lever for reducing errors on proper nouns and jargon.\n"
)

JARGON_TEXT = (
    "We're evaluating Qdrant versus Weaviate for our vector store. "
    "The gpt-5.6-luna embeddings fit in our 1.05M context window comfortably."
)

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
    audio_path2 = tf.name

try:
    _generate_test_audio(JARGON_TEXT, audio_path2)

    # Without hints
    plain = _transcribe("gpt-transcribe", audio_path2)
    print(f"Without hints:\n  {plain['text']}\n")

    # With context + keyword hints
    hinted = _transcribe(
        "gpt-transcribe",
        audio_path2,
        prompt="Technical discussion about vector databases and OpenAI models.",
        keywords=["Qdrant", "Weaviate", "gpt-5.6-luna", "1.05M"],
    )
    print(f"With hints:\n  {hinted['text']}\n")

finally:
    os.unlink(audio_path2)


# ---- Example 3: Verbose JSON — timestamps and language detection -----------

print("=" * 60)
print("EXAMPLE 3: verbose_json — segments with timestamps")
print("=" * 60)
print()

MULTI_SENTENCE = (
    "Good morning. Today we discuss the Responses API migration. "
    "Three teams are affected: platform, integrations, and billing."
)

with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
    audio_path3 = tf.name

try:
    _generate_test_audio(MULTI_SENTENCE, audio_path3)

    with open(audio_path3, "rb") as f:
        verbose = client.audio.transcriptions.create(
            model="gpt-transcribe",
            file=f,
            response_format="verbose_json",
        )

    print(f"Detected language: {verbose.language}")
    print(f"Duration:          {verbose.duration:.1f}s")
    print(f"Full text:         {verbose.text}\n")

    if verbose.segments:
        print("Segments:")
        for seg in verbose.segments[:4]:
            print(f"  [{seg.start:.2f}s – {seg.end:.2f}s]  {seg.text.strip()}")

finally:
    os.unlink(audio_path3)


# ---- Summary ----------------------------------------------------------------

print()
print("=" * 60)
print("TRANSCRIPTION MODEL REFERENCE")
print("=" * 60)
print("""
Model comparison (July 2026):

  Model                  Pricing        Use case
  ─────────────────────  ─────────────  ────────────────────────────────────
  whisper-1              $0.006/min     Legacy; still available, not updated
  gpt-4o-transcribe      $0.006/min     Previous recommended default
  gpt-transcribe         $0.0045/min    New default. 25% cheaper + ~half WER
  gpt-live-transcribe    $0.017/min     Real-time streaming (WebSocket)

Key params for client.audio.transcriptions.create():
  model              — "gpt-transcribe" (file) or "gpt-live-transcribe" (streaming)
  file               — Audio file object (mp3, wav, m4a, flac, ogg, webm, …)
  prompt             — Free-form context string (domain, speaker names, jargon)
  keywords           — List[str] of terms to recognise exactly
  language           — BCP-47 language code hint (e.g. "en", "es", "fr")
  expected_input_languages — List[str] for multilingual audio
  response_format    — "json" (default) | "verbose_json" | "diarized_json" | "text" | "vtt" | "srt"
  timestamp_granularities — ["word"] and/or ["segment"] for verbose_json

gpt-live-transcribe (streaming):
  Uses the same WebSocket session pattern as gpt-realtime-whisper (see ex. 33).
  Send session.update with {"model": "gpt-live-transcribe"}, stream audio chunks,
  receive conversation.item.input_audio_transcription.delta events.

Batch API:
  gpt-transcribe supports the Batch API (client.batches.*) for 50% cost reduction
  on async workloads where latency is not a concern.

Migration from whisper-1 / gpt-4o-transcribe:
  Drop-in: change the model string. No other parameters change.
  Use gpt-transcribe as the new default for all new file-transcription work.
""")

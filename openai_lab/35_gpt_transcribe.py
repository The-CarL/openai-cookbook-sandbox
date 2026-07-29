"""Exercise 35: GPT-Transcribe and GPT-Live-Transcribe (July 28, 2026).

Two new transcription models that replace whisper-1 for most workloads:
  gpt-transcribe      — async file transcription. $0.0045/min of audio.
  gpt-live-transcribe — WebSocket streaming for real-time sessions. $0.017/min.

Both support free-form prompt context, keyword hints, and multi-language hints.
On Common Voice (22 languages), WER drops from 40.4% (whisper-1) to 19.3% (gpt-transcribe).
"""

import io
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


# --- Step 0: Generate sample audio using TTS so the exercise is self-contained ---
print("=" * 60)
print("SETUP: Generating sample audio via TTS")
print("=" * 60)
print()

SAMPLE_TEXT = (
    "The Q3 revenue was $4.2 million, up 18% year-over-year. "
    "Our NPS score improved to 72, and churn dropped to 1.4%. "
    "The main driver was the EMEA expansion — especially Germany and the Netherlands."
)

tts_response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=SAMPLE_TEXT,
    response_format="mp3",
)
audio_bytes = tts_response.content
audio_duration_estimate = len(SAMPLE_TEXT.split()) / 150  # rough words-per-minute
print(f"Generated {len(audio_bytes):,} bytes of audio (~{audio_duration_estimate:.1f} min).")
print()


# --- Example 1: whisper-1 baseline ---
print("=" * 60)
print("EXAMPLE 1: whisper-1 (baseline)")
print("=" * 60)
print()

start = time.time()
t_whisper = client.audio.transcriptions.create(
    model="whisper-1",
    file=("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg"),
)
whisper_elapsed = time.time() - start

print(f"Transcript: {t_whisper.text}")
print(f"Latency:    {whisper_elapsed:.2f}s")
print()


# --- Example 2: gpt-transcribe, no hints ---
print("=" * 60)
print("EXAMPLE 2: gpt-transcribe — no context hints")
print("=" * 60)
print()

start = time.time()
t_basic = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg"),
)
basic_elapsed = time.time() - start

print(f"Transcript: {t_basic.text}")
print(f"Latency:    {basic_elapsed:.2f}s")
print()


# --- Example 3: gpt-transcribe with context + keyword hints ---
print("=" * 60)
print("EXAMPLE 3: gpt-transcribe — with prompt context + keywords")
print("=" * 60)
print()
print("Context helps with jargon, acronyms, and proper nouns.")
print()

start = time.time()
t_hinted = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg"),
    prompt=(
        "This is a sales QBR (quarterly business review) in English. "
        "The speaker uses standard SaaS metrics like NPS, ARR, and churn."
    ),
    keywords=["Q3", "NPS", "EMEA", "Netherlands"],
    language="en",
)
hinted_elapsed = time.time() - start

print(f"Transcript: {t_hinted.text}")
print(f"Latency:    {hinted_elapsed:.2f}s")
print()


# --- Example 4: JSON response_format for language detection ---
print("=" * 60)
print("EXAMPLE 4: gpt-transcribe — verbose_json with language detection")
print("=" * 60)
print()

t_verbose = client.audio.transcriptions.create(
    model="gpt-transcribe",
    file=("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg"),
    response_format="verbose_json",
)
print(f"Detected language: {getattr(t_verbose, 'language', 'n/a')}")
print(f"Duration:          {getattr(t_verbose, 'duration', 'n/a')}s")
print(f"Transcript:        {t_verbose.text}")
print()


# --- Summary ---
print("=" * 60)
print("SUMMARY")
print("=" * 60)

cost_per_min_whisper = 0.006   # whisper-1
cost_per_min_gpt     = 0.0045  # gpt-transcribe
dur_min = audio_duration_estimate

print(f"""
Model comparison (same audio, ~{dur_min:.2f} min):
  whisper-1:      {whisper_elapsed:.2f}s latency, ~${cost_per_min_whisper * dur_min:.5f} cost
  gpt-transcribe: {basic_elapsed:.2f}s latency, ~${cost_per_min_gpt * dur_min:.5f} cost

gpt-transcribe is 25% cheaper than whisper-1 and dramatically more
accurate on real-world audio (19.3% vs 40.4% WER on Common Voice 22).

Key parameters:
  prompt    — free-form context about the recording domain / speaker
  keywords  — list of literal terms that may appear (proper nouns, codes)
  language  — ISO-639-1 hint; omit to let the model auto-detect

Response formats:
  text / json / srt / vtt — same as whisper-1
  verbose_json            — adds language, duration, word-level segments

Pricing:
  gpt-transcribe:      $0.0045 / min of audio (file upload)
  gpt-live-transcribe: $0.017  / min of session audio (WebSocket streaming)
  whisper-1:           $0.006  / min (kept for batch compatibility)

gpt-live-transcribe uses the same WebSocket protocol as the Realtime API
(see ex. 33). It's designed for low-latency committed-turn transcription —
voice assistants, live captions, call-center agents. Not shown here because
it requires a WebSocket session; see the Realtime API docs for the pattern.
""")

#!/usr/bin/env python3
"""
arey_variants.py — Test how faster-whisper transcribes Hindi "arey/aray"

Run this script, say "arey" (and variants) into the mic a few times,
then check the [TRANSCRIPT] output to see exactly what Whisper hears.
Add any new variants to the AREY_VARIANTS list below.

Usage:
    python arey_variants.py
"""

import re
import sys
import queue
import threading
import time

missing = []
try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    missing.append("sounddevice")

try:
    from faster_whisper import WhisperModel
except ImportError:
    missing.append("faster-whisper")

if missing:
    print(f"[ERROR] Missing: {', '.join(missing)}")
    print(f"        pip install {' '.join(missing)}")
    sys.exit(1)

# ── Known variations ──────────────────────────────────────────────────────────
# How Whisper (English mode) might transcribe Hindi "अरे" (arey/aray):
#   - Spoken form      → likely Whisper output
#   - arey  (short e)  → "arey", "are", "array", "a ray", "Ari"
#   - aray  (long a)   → "array", "a ray", "aray"
#   - arrey (doubled r)→ "array", "are"
#   - are yaar         → "are ya", "are you"

AREY_VARIANTS_ROMAN = [
    "arey",   # direct romanized match
    "aray",   # alternate spelling
    "arrey",  # doubled-r version
    "array",  # confirmed Whisper English mishear
]

AREY_VARIANTS_SCRIPT = [
    "अरे",    # Hindi Devanagari
    "अरेय",   # Hindi Devanagari variant
    "あれ",    # Japanese hiragana — confirmed Whisper maps "arey" here
    "アレ",    # Japanese katakana variant
]

AREY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in AREY_VARIANTS_ROMAN) + r")\b"
    r"|a\s+ray"       # "a ray" split mishear
    r"|ar\s+e[.\s]"   # "Ar e." split mishear
    r"|" + "|".join(re.escape(v) for v in AREY_VARIANTS_SCRIPT),
    re.IGNORECASE
)

ALL_VARIANTS = AREY_VARIANTS_ROMAN + AREY_VARIANTS_SCRIPT

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 4
MODEL_SIZE    = "base"

# ── Audio + transcription ─────────────────────────────────────────────────────

def audio_capture(audio_queue: queue.Queue, stop_event: threading.Event):
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS
    buf = []

    def callback(indata, frames, time_info, status):
        buf.append(indata.copy())
        if sum(len(x) for x in buf) >= chunk_samples:
            combined = np.concatenate(buf).flatten().astype(np.float32) / 32768.0
            audio_queue.put(combined)
            buf.clear()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        callback=callback):
        while not stop_event.is_set():
            time.sleep(0.05)


def main():
    print(f"Loading model '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("Ready.\n")
    print("Say 'arey' or 'aray' into the mic. Ctrl+C to stop.")
    print("Watch [TRANSCRIPT] to see exactly what Whisper hears.\n")
    print(f"Matching against: {', '.join(ALL_VARIANTS)}\n")
    print("-" * 50)

    audio_q    = queue.Queue()
    stop_event = threading.Event()
    count      = 0

    t = threading.Thread(target=audio_capture, args=(audio_q, stop_event), daemon=True)
    t.start()

    try:
        while True:
            audio = audio_q.get()
            segments, _ = model.transcribe(
                audio,
                language=None,   # auto-detect: allows Hindi segments
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if not text:
                continue

            matches = AREY_PATTERN.findall(text)
            print(f"[TRANSCRIPT] {text}")
            if matches:
                count += len(matches)
                print(f"  ✓ AREY detected! matched: {matches}  (total: {count})")
            print()

    except KeyboardInterrupt:
        stop_event.set()
        print(f"\nTotal 'arey' detected: {count}")
        print("\nIf Whisper used a word NOT in the list above, add it to AREY_VARIANTS.")


if __name__ == "__main__":
    main()

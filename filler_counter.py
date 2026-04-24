#!/usr/bin/env python3
"""
Real-time filler word counter — faster-whisper edition
Tracks: uh, um, ah, er from microphone (handles accented English)

Requirements:
    pip install faster-whisper sounddevice numpy rich

Model (~150MB for 'base') downloads automatically on first run.
"""

import re
import sys
import threading
import time
import queue
from collections import defaultdict, deque

# ── Dependency checks ─────────────────────────────────────────────────────────

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

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── Config ────────────────────────────────────────────────────────────────────

FILLERS = ["uh", "um", "ah", "er", "okay"]

# Match plain, repeated, and whisper-parenthesized forms: uh, uhh, (uh), [uh]
PATTERNS = {
    "uh":   re.compile(r"[\[\(]?u+h+[\]\)]?", re.IGNORECASE),
    "um":   re.compile(r"[\[\(]?u+m+[\]\)]?", re.IGNORECASE),
    "ah":   re.compile(r"[\[\(]?a+h+[\]\)]?", re.IGNORECASE),
    "er":   re.compile(r"[\[\(]?e+r+[\]\)]?", re.IGNORECASE),
    "okay": re.compile(r"\b(ok|okay)\b", re.IGNORECASE),
}

FILLER_COLORS = {
    "uh":   "bright_red",
    "um":   "orange1",
    "ah":   "bright_yellow",
    "er":   "cyan",
    "okay": "bright_magenta",
}

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 4      # seconds of audio per transcription pass
MODEL_SIZE    = "base" # tiny | base | small  (base = best speed/accuracy balance)
LOG_MAX       = 6

# ── Counter state ─────────────────────────────────────────────────────────────

class FillerCounter:
    def __init__(self):
        self.counts = defaultdict(int)
        self.log: deque = deque(maxlen=LOG_MAX)
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.last_transcript = ""

    def process(self, text: str):
        text = text.strip()
        if not text:
            return
        self.last_transcript = text
        found = {}
        for filler, pattern in PATTERNS.items():
            n = len(pattern.findall(text))
            if n:
                found[filler] = n
        if found:
            with self._lock:
                for f, n in found.items():
                    self.counts[f] += n
            self.log.append((text, found))

    def snapshot(self):
        with self._lock:
            return dict(self.counts)

    def total(self):
        with self._lock:
            return sum(self.counts.values())

    def elapsed(self):
        return time.time() - self.start_time


# ── Rich display ──────────────────────────────────────────────────────────────

def build_panel(counter: FillerCounter, status: str) -> Panel:
    counts  = counter.snapshot()
    total   = counter.total()
    elapsed = counter.elapsed()
    mins, secs = divmod(int(elapsed), 60)

    out = Text()

    # Counter row
    out.append("\n  ")
    for filler in FILLERS:
        color = FILLER_COLORS[filler]
        n = counts.get(filler, 0)
        out.append(f" {filler.upper()} ", style=f"bold {color} on grey11")
        out.append(f" {n:<3}", style="bold white")
        out.append("   ")
    out.append("│  TOTAL ", style="bold white")
    out.append(f"{total}", style="bold green")
    out.append(f"   {mins:02d}:{secs:02d}", style="dim")
    out.append("\n")

    # Rate
    if elapsed > 5 and total > 0:
        rate = total / (elapsed / 60)
        out.append(f"\n  Rate: {rate:.1f} fillers/min\n", style="dim")
    else:
        out.append("\n")

    # Transcript log
    if counter.log:
        out.append("  ─── Recent transcript ───────────────────────\n", style="dim")
        for transcript, found in list(counter.log):
            out.append("  ")
            for word in transcript.split():
                clean = re.sub(r"[^a-z]", "", word.lower())
                if clean in FILLER_COLORS:
                    out.append(word, style=f"bold {FILLER_COLORS[clean]}")
                else:
                    out.append(word, style="white")
                out.append(" ")
            out.append("\n")

    # Status line
    out.append(f"\n  ▶ {status}", style="dim italic")

    return Panel(
        out,
        title="[bold blue] Filler Word Counter [/bold blue]",
        subtitle="[dim]Ctrl+C to stop[/dim]",
        border_style="blue",
        padding=(0, 1),
    )


# ── Threads ───────────────────────────────────────────────────────────────────

def audio_capture(audio_queue: queue.Queue, stop_event: threading.Event):
    """Accumulate CHUNK_SECONDS of audio then push float32 array to queue."""
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


def transcriber(model: WhisperModel, audio_queue: queue.Queue,
                counter: FillerCounter, status_ref: list,
                stop_event: threading.Event):
    """Pull audio chunks, transcribe, update counter."""
    while not stop_event.is_set():
        try:
            audio = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        status_ref[0] = "transcribing..."
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            counter.process(text)
            status_ref[0] = text[:80]
        else:
            status_ref[0] = "(silence)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading faster-whisper '{MODEL_SIZE}' model...")
    print("(downloads ~150MB on first run)\n")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    try:
        dev = sd.query_devices(kind="input")
        print(f"Microphone : {dev['name']}")
    except Exception:
        print("Microphone : (default device)")

    print(f"\nTracking   : {' | '.join(f.upper() for f in FILLERS)}")
    print(f"Chunk size : {CHUNK_SECONDS}s per transcription pass")
    print("Listening  : Press Ctrl+C to stop.\n")

    counter    = FillerCounter()
    status_ref = ["waiting for audio..."]
    stop_event = threading.Event()
    audio_q    = queue.Queue()

    t_audio = threading.Thread(
        target=audio_capture, args=(audio_q, stop_event), daemon=True
    )
    t_trans = threading.Thread(
        target=transcriber, args=(model, audio_q, counter, status_ref, stop_event),
        daemon=True
    )
    t_audio.start()
    t_trans.start()

    if HAS_RICH:
        console = Console()
        with Live(build_panel(counter, status_ref[0]), console=console,
                  refresh_per_second=4, transient=False) as live:
            try:
                while True:
                    live.update(build_panel(counter, status_ref[0]))
                    time.sleep(0.25)
            except KeyboardInterrupt:
                pass
    else:
        try:
            while True:
                counts = counter.snapshot()
                parts  = [f"{f.upper()}:{counts.get(f, 0)}" for f in FILLERS]
                parts.append(f"TOTAL:{counter.total()}")
                print(f"\r{'  '.join(parts)}", end="", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    stop_event.set()

    # Final summary
    elapsed    = counter.elapsed()
    mins, secs = divmod(int(elapsed), 60)
    counts     = counter.snapshot()
    total      = counter.total()

    print("\n\n╔══════════════════════════╗")
    print("║   Final Counts           ║")
    print("╠══════════════════════════╣")
    for filler in FILLERS:
        print(f"║  {filler.upper():<6}  {counts.get(filler, 0):>4}            ║")
    print("╠══════════════════════════╣")
    print(f"║  TOTAL   {total:>4}            ║")
    print(f"║  Time    {mins:02d}:{secs:02d}            ║")
    if elapsed > 5 and total > 0:
        rate = total / (elapsed / 60)
        print(f"║  Rate    {rate:>5.1f}/min       ║")
    print("╚══════════════════════════╝")


if __name__ == "__main__":
    main()

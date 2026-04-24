# Filler Word Counter

Real-time filler word detector using your microphone. Counts **uh, um, ah, er, okay** as you speak — live, offline, no playback needed.

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition and [Rich](https://github.com/Textualize/rich) for the terminal display.

---

## Features

- Real-time mic input — no recording or playback required
- Tracks: **UH | UM | AH | ER | OKAY**
- Live terminal display with per-word counts, elapsed time, and fillers/min rate
- Transcript log with filler words highlighted in colour
- Final summary on exit
- Fully offline after model download

---

## Requirements

- Python 3.10+
- Windows / macOS / Linux

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/akashabbigeri/filler-word-counter.git
cd filler-word-counter
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> The Whisper `base` model (~150MB) downloads automatically on first run.

---

## Usage

```bash
python filler_counter.py
```

Press `Ctrl+C` to stop. A final count summary is printed on exit.

---

## How It Works

1. Microphone audio is captured via `sounddevice` in 4-second chunks
2. Each chunk is transcribed by `faster-whisper` (offline, CPU)
3. Transcripts are scanned for filler words using regex
4. Counts update live in the terminal via `Rich`

Filler words are only counted from **finalised** transcript segments to avoid double-counting.

---

## Bonus: Hindi "Arey" Detection

`arey_variants.py` is a separate test script for detecting the Hindi filler word **"arey/aray"**. Because Whisper maps it inconsistently across languages (English, Japanese, Hindi script), run this script to observe how your voice is transcribed and tune the patterns accordingly.

```bash
python arey_variants.py
```

---

## Configuration

Edit the top of `filler_counter.py` to adjust:

| Variable | Default | Description |
|---|---|---|
| `FILLERS` | `["uh","um","ah","er","okay"]` | Words to track |
| `MODEL_SIZE` | `"base"` | Whisper model: `tiny` / `base` / `small` |
| `CHUNK_SECONDS` | `4` | Seconds of audio per transcription pass |

# Realtime Voice Agent

Low-latency speech-to-speech voice agent targeting **< 1 second ear-to-mouth** latency. Built with async streaming through STT, LLM, and TTS, a 2-word early dispatch trigger, and a persistent HTTP/2 connection pool that eliminates cold-start overhead.

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Microphone  │──▶│  WebRTC VAD  │──▶│  STT Module  │
│   (PyAudio)  │   │  (Aggress: 3)│   │   (wav2vec2) │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                             │
                            ┌────────────────┘
                            ▼
            ┌────────────────────────────────────────┐
            │   LLM (OpenAI-compatible, streaming)   │
            │   SSE + 2-word trigger dispatch         │
            └────────────────┬───────────────────────┘
                             │ first 2 words → TTS dispatch
                             ▼
            ┌────────────────────────────────────────┐
            │   TTS (multiple providers supported)   │
            │   Streaming sub-chunks (24 kHz PCM)    │
            └────────────────┬───────────────────────┘
                             ▼
                   ┌──────────────────┐
                   │ AudioPlaybackQueue│  ordered by chunk_id
                   └─────────┬────────┘
                             ▼
                   ┌──────────────────┐
                   │     Speaker      │
                   └──────────────────┘
```

### Key Design Decisions

- **Global connection pool** — httpx clients are created at import and never closed. Eliminates ~1.5s of TLS handshake per request.
- **2-word trigger** — TTS dispatch begins after just 2 LLM tokens; subsequent chunks every 7 words. Punctuation breaks earlier and prevents 1–2 word orphans.
- **Streaming TTS playback** — providers stream raw PCM; `AudioPlaybackQueue` plays sub-chunks in order, buffering by `chunk_id` so out-of-order arrivals still play correctly.
- **Turn serialization** — in `--no-interrupt` mode, a new turn waits for the previous one to finish; chat history is normalized to maintain correct message alternation even after interruptions.

### Modules

| File | Role |
|------|------|
| `main_client.py` | Orchestrator: VAD capture, turn handling, ordered playback, latency logging |
| `connection_pool.py` | Global httpx clients for STT / LLM / TTS; connection warming |
| `stt_module.py` | wav2vec2 transcription (OpenAI-compatible API) |
| `llm_module.py` | Streaming LLM with smart word chunking and normalizing chat history |
| `tts_module.py` | Streaming TTS dispatcher (raw 24 kHz PCM sub-chunks) |
| `prompt_config.py` | System prompt configuration |
| `api_keys.py` | Reads endpoints + API keys from `.env` |

---

## Setup

### Prerequisites

- Python 3.10–3.12
- `portaudio` (for PyAudio):
  - macOS: `brew install portaudio`
  - Debian/Ubuntu: `apt install portaudio19-dev`

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `webrtcvad` fails to build:
```bash
pip install webrtcvad-wheels
```

### Configure

```bash
cp .env.example .env
# Fill in your STT / LLM / TTS endpoints and API keys
```

`.env` is gitignored — never commit it.

---

## Running

```bash
# Defaults: primary LLM + TTS provider A, default scenario, barge-in enabled
python main_client.py
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--gemma` / `--gemma4` / `--gemma4-think` / `--gpt` | LLM provider (mutually exclusive) | `--gemma` |
| `-t`, `--tts` | TTS provider: `azure` / `azure-speech` / `elevenlabs` | `azure` |
| `-s`, `--scenario` | Scenario name from `prompt_config.py` | `default` |
| `--allow-interrupt` / `--no-interrupt` | Barge-in toggle | barge-in on |

### Examples

```bash
# Secondary LLM, ElevenLabs TTS, no barge-in
python main_client.py --gemma4 -t elevenlabs --no-interrupt

# GPT endpoint with cognitive services TTS
python main_client.py --gpt -t azure-speech

# Secondary LLM with reasoning/thinking mode
python main_client.py --gemma4-think -t elevenlabs --no-interrupt
```

Press **Ctrl+C** to exit. The bot also auto-exits when the LLM emits `[HANGUP]`.

### Barge-in

In `--allow-interrupt` mode, VAD keeps listening while the bot speaks — saying anything stops playback and starts a new turn. **Requires headphones** to avoid the bot interrupting itself.

In `--no-interrupt` mode, VAD is muted during playback. No headphones needed; conversation is strictly turn-taking.

---

## Supported Provider Types

The system is designed to work with any OpenAI-compatible API endpoints:

| Layer | Requirements |
|-------|-------------|
| **STT** | OpenAI-compatible `/v1/audio/transcriptions` endpoint |
| **LLM** | OpenAI-compatible `/v1/chat/completions` with streaming SSE |
| **TTS** | OpenAI-compatible TTS REST, Cognitive Services Speech REST, or ElevenLabs API |

All credentials are loaded from `.env` via `api_keys.py`.

---

## Audio Format

- **Mic capture**: 16 kHz, mono, 16-bit (WebRTC VAD requirement)
- **TTS output**: 24 kHz, mono, 16-bit raw PCM

---

## Configuration Constants

`main_client.py`:
```python
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
VAD_AGGRESSIVENESS = 3
SILENCE_THRESHOLD_MS = 500
MIN_SPEECH_DURATION_MS = 300
```

`llm_module.py`:
```python
FIRST_TRIGGER_WORDS = 2
CHUNK_WORD_SIZE = 7
```

---

## Latency Budget

```
VAD → STT send         <  10 ms
STT inference          < 500 ms
LLM TTFT               < 300 ms
Glue (TTFT → 2 words)  < 200 ms
TTS TTFB               ~ 150 ms
─────────────────────────────────
Ear-to-mouth target    < 1000 ms
```

The 2-word trigger dispatches the first TTS chunk before the LLM has finished generating, so TTS TTFB overlaps with the LLM's tail-end generation.

---

## Project Layout

```
├── main_client.py         # entrypoint
├── connection_pool.py
├── stt_module.py
├── llm_module.py
├── tts_module.py
├── prompt_config.py
├── api_keys.py            # reads .env
├── requirements.txt
├── .env.example
├── .env                   # gitignored
└── .gitignore
```

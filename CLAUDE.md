# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python voice assistant targeting <1s ear-to-mouth latency. Async pipeline with persistent HTTP connection pooling, WebRTC VAD, streaming LLM with a 2-word trigger that dispatches the first TTS chunk before the LLM is finished.

LLM and TTS providers are selected via command-line flags.

## Commands

```bash
# Setup (macOS)
brew install portaudio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run (defaults: primary LLM + TTS provider A, scenario=default)
python main_client.py

# LLM:       --gemma | --gemma4 | --gemma4-think | --gpt
# TTS:       --tts azure | --tts azure-speech | --tts elevenlabs
# Scenario:  -s default
# Barge-in:  --allow-interrupt (default) | --no-interrupt
```

Press `Ctrl+C` to exit. The bot also auto-exits when the LLM emits `[HANGUP]`.

## Architecture

```
Microphone → WebRTC VAD → STT (wav2vec2) → LLM (streaming SSE)
                                              ↓ (2-word trigger)
Speaker ← AudioPlaybackQueue ← TTS (streaming PCM sub-chunks)
```

### Modules

| File | Purpose |
|------|---------|
| `main_client.py` | Orchestrator: VAD capture, turn handling, ordered playback queue, latency logging |
| `connection_pool.py` | Global httpx clients for STT / LLM / TTS; warming |
| `llm_module.py` | Streaming LLM with smart word chunking and a normalising chat history |
| `tts_module.py` | Streaming TTS dispatcher (raw 24 kHz PCM sub-chunks) |
| `stt_module.py` | wav2vec2 transcription |
| `prompt_config.py` | System prompt configuration |
| `api_keys.py` | Endpoints + API keys from `.env` (gitignored) |

### Key Design Patterns

- **Global connection pool**: clients are created at import and never closed. Eliminates ~1.5 s of TLS handshake per request.
- **2-word trigger**: TTS dispatch begins after `FIRST_TRIGGER_WORDS` (default 2); subsequent chunks every `CHUNK_WORD_SIZE` (default 7). Punctuation breaks earlier and prevents 1-2-word orphans.
- **Streaming TTS playback**: providers stream raw PCM; `AudioPlaybackQueue` plays sub-chunks in order, buffering by `chunk_id` so out-of-order arrivals still play correctly.
- **TTS pre-warm**: while STT runs, `prewarm_tts_connection()` runs concurrently.
- **Turn serialisation**: in `--no-interrupt` mode, a new turn waits for the previous one to finish; `ChatHistory._normalize()` collapses any out-of-order user/assistant entries before sending to the API so interruptions don't break alternation.

### Configuration Constants

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

### Audio Format

- Mic capture: 16 kHz, mono, 16-bit (WebRTC VAD requirement)
- TTS output: 24 kHz, mono, 16-bit raw PCM

### Provider Types

| Layer | Type | Notes |
|-------|------|-------|
| STT   | OpenAI-compatible | wav2vec2 endpoint |
| LLM   | OpenAI-compatible | primary endpoint (default) |
| LLM   | OpenAI-compatible | secondary endpoint (`--gemma4`/`--gemma4-think`) |
| LLM   | OpenAI-compatible | GPT endpoint (`--gpt`); uses `max_completion_tokens` |
| TTS   | OpenAI-compatible | REST TTS with HTTP/2 streaming |
| TTS   | Cognitive Services | Speech REST with SSML |
| TTS   | ElevenLabs | streaming PCM |

API credentials live in `.env` (gitignored).

## Latency Budget Target

```
VAD → STT send         <  10 ms
STT inference          < 500 ms
LLM TTFT               < 300 ms
Glue (TTFT → 2 words)  < 200 ms
TTS TTFB               ~ 150 ms
─────────────────────────────────
Ear-to-mouth target    < 1000 ms
```

# -*- coding: utf-8 -*-
"""
stt_module.py - Speech-to-Text via wav2vec2 (OpenAI-compatible endpoint).

Wraps raw PCM audio in a WAV container and posts to the STT API using the
shared connection pool.
"""

import io
import time
import wave
from dataclasses import dataclass
from typing import Optional

from connection_pool import STT_CLIENT, STT_MODEL, get_stt_headers


@dataclass
class STTResult:
    text: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


async def transcribe_audio(
    audio_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> STTResult:
    """Transcribe raw PCM audio to text."""
    start = time.time()
    try:
        wav_bytes = _wrap_pcm_as_wav(audio_data, sample_rate, channels, sample_width)
        response = await STT_CLIENT.post(
            "/v1/audio/transcriptions",
            data={"model": STT_MODEL},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            headers=get_stt_headers(),
        )
        latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            return STTResult(
                text=response.json().get("text", "").strip(),
                latency_ms=latency_ms,
                success=True,
            )
        return STTResult(
            text="",
            latency_ms=latency_ms,
            success=False,
            error=f"HTTP {response.status_code}: {response.text}",
        )
    except Exception as e:
        return STTResult(
            text="",
            latency_ms=(time.time() - start) * 1000,
            success=False,
            error=str(e),
        )

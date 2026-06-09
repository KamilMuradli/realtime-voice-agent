# -*- coding: utf-8 -*-
"""
tts_module.py - Streaming Text-to-Speech (Azure OpenAI / ElevenLabs).

Streams raw 24kHz 16-bit mono PCM sub-chunks via on_audio_chunk so playback
can begin before synthesis completes. Uses the global httpx clients from
connection_pool.
"""

import io
import wave
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, List, Optional

from connection_pool import (
    TTS_CLIENT,
    TTS_ENDPOINT,
    TTS_VOICE,
    get_tts_headers,
    ensure_tts_warm,
    AZURE_SPEECH_CLIENT,
    AZURE_SPEECH_ENDPOINT,
    AZURE_SPEECH_VOICE,
    get_azure_speech_headers,
    ensure_azure_speech_warm,
    ELEVENLABS_CLIENT,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    get_elevenlabs_headers,
    ensure_elevenlabs_warm,
)


class TTSProvider(Enum):
    AZURE = "azure"
    AZURE_SPEECH = "azure-speech"
    ELEVENLABS = "elevenlabs"


ACTIVE_TTS_PROVIDER: TTSProvider = TTSProvider.ELEVENLABS


def set_tts_provider(provider: TTSProvider) -> None:
    global ACTIVE_TTS_PROVIDER
    ACTIVE_TTS_PROVIDER = provider
    print(f"   🔊 TTS Provider set to: {provider.value}")


def get_tts_provider() -> TTSProvider:
    return ACTIVE_TTS_PROVIDER


@dataclass
class TTSResult:
    audio_data: bytes
    ttfb_ms: float
    total_latency_ms: float
    audio_bytes: int
    success: bool
    provider: str = ""
    error: Optional[str] = None


OnAudioChunk = Callable[[bytes, bool, bool], Awaitable[None]]


def _azure_request(text: str, voice: Optional[str]):
    return TTS_CLIENT, TTS_ENDPOINT, {
        "json": {
            "model": "tts-1",
            "input": text,
            "voice": voice or TTS_VOICE,
            "response_format": "pcm",
        },
        "headers": get_tts_headers(),
    }


def _azure_speech_request(text: str, voice: Optional[str]):
    voice_name = voice or AZURE_SPEECH_VOICE
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    ssml = (
        f'<speak version="1.0" xml:lang="en-US">'
        f'<voice name="{voice_name}">{safe}</voice></speak>'
    )
    return AZURE_SPEECH_CLIENT, AZURE_SPEECH_ENDPOINT, {
        "content": ssml.encode("utf-8"),
        "headers": get_azure_speech_headers(),
    }


def _elevenlabs_request(text: str, voice: Optional[str]):
    voice_id = voice or ELEVENLABS_VOICE_ID
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "language_code": "az",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    return ELEVENLABS_CLIENT, f"/v1/text-to-speech/{voice_id}/stream", {
        "json": payload,
        "params": {"output_format": "pcm_24000"},
        "headers": get_elevenlabs_headers(),
    }


async def _stream_pcm(
    provider: TTSProvider,
    text: str,
    on_audio_chunk: Optional[OnAudioChunk],
    voice: Optional[str],
) -> TTSResult:
    """Shared streaming core. Buffers odd trailing bytes to keep 16-bit samples aligned."""
    if provider == TTSProvider.AZURE:
        client, endpoint, kwargs = _azure_request(text, voice)
        provider_tag = "azure-stream"
    elif provider == TTSProvider.AZURE_SPEECH:
        client, endpoint, kwargs = _azure_speech_request(text, voice)
        provider_tag = "azure-speech-stream"
    else:
        client, endpoint, kwargs = _elevenlabs_request(text, voice)
        provider_tag = "elevenlabs-stream"

    start = time.time()
    ttfb: Optional[float] = None
    all_audio = bytearray()
    leftover = b""
    first_sent = False

    try:
        async with client.stream("POST", endpoint, **kwargs) as response:
            if response.status_code != 200:
                err = (await response.aread()).decode(errors="replace")
                return TTSResult(
                    audio_data=b"", ttfb_ms=0,
                    total_latency_ms=(time.time() - start) * 1000,
                    audio_bytes=0, success=False, provider=provider_tag,
                    error=f"HTTP {response.status_code}: {err}",
                )

            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if ttfb is None:
                    ttfb = time.time()

                data = leftover + chunk
                leftover = b""
                if len(data) % 2:
                    leftover = data[-1:]
                    data = data[:-1]
                if not data:
                    continue

                all_audio.extend(data)
                is_first = not first_sent
                first_sent = True
                if on_audio_chunk:
                    await on_audio_chunk(data, is_first, False)

        if on_audio_chunk and first_sent:
            await on_audio_chunk(b"", False, True)

        end = time.time()
        return TTSResult(
            audio_data=bytes(all_audio),
            ttfb_ms=(ttfb - start) * 1000 if ttfb else 0,
            total_latency_ms=(end - start) * 1000,
            audio_bytes=len(all_audio),
            success=True,
            provider=provider_tag,
        )

    except Exception as e:
        return TTSResult(
            audio_data=b"", ttfb_ms=0,
            total_latency_ms=(time.time() - start) * 1000,
            audio_bytes=0, success=False, provider=provider_tag, error=str(e),
        )


async def synthesize_speech_streaming(
    text: str,
    on_audio_chunk: Optional[OnAudioChunk] = None,
    voice: Optional[str] = None,
    provider: Optional[TTSProvider] = None,
) -> TTSResult:
    """Stream synthesized PCM audio for the active (or specified) provider."""
    return await _stream_pcm(provider or ACTIVE_TTS_PROVIDER, text, on_audio_chunk, voice)


async def prewarm_tts_connection(provider: Optional[TTSProvider] = None) -> None:
    provider = provider or ACTIVE_TTS_PROVIDER
    if provider == TTSProvider.AZURE:
        await ensure_tts_warm()
    elif provider == TTSProvider.AZURE_SPEECH:
        await ensure_azure_speech_warm()
    elif provider == TTSProvider.ELEVENLABS:
        await ensure_elevenlabs_warm()


def pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    """Wrap raw PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


def merge_wav_chunks(chunks: List[bytes]) -> bytes:
    """Concatenate WAV chunks into a single WAV. Falls back to raw concat on error."""
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    try:
        with wave.open(io.BytesIO(chunks[0]), "rb") as wav:
            params = wav.getparams()
    except Exception:
        return b"".join(chunks)

    frames: List[bytes] = []
    for chunk in chunks:
        try:
            with wave.open(io.BytesIO(chunk), "rb") as wav:
                frames.append(wav.readframes(wav.getnframes()))
        except Exception:
            continue

    out = io.BytesIO()
    with wave.open(out, "wb") as wav_out:
        wav_out.setnchannels(params.nchannels)
        wav_out.setsampwidth(params.sampwidth)
        wav_out.setframerate(params.framerate)
        for f in frames:
            wav_out.writeframes(f)
    return out.getvalue()

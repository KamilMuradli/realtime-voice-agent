# -*- coding: utf-8 -*-
"""
connection_pool.py - Global httpx clients (STT, LLM, TTS).

Clients are created once at import time and never closed for the lifetime of
the process. This eliminates the per-request TLS handshake (~1.5s cold start).
"""

import asyncio
import io
import time
import wave
from typing import Awaitable, Callable, Dict

import httpx

from api_keys import (
    STT_BASE_URL, STT_API_KEY,
    LLM_BASE_URL, LLM_API_KEY,
    GEMMA_SC_BASE_URL,
    GPT_BASE_URL, GPT_API_KEY,
    TTS_BASE_URL, TTS_API_KEY,
    AZURE_SPEECH_REGION, AZURE_SPEECH_KEY,
    ELEVENLABS_API_KEY,
)


_LIMITS = httpx.Limits(
    max_keepalive_connections=100,
    max_connections=100,
    keepalive_expiry=None,  # never expire keep-alive connections
)
_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)


# --- Provider configuration ------------------------------------------------

STT_MODEL = "openai/wav2vec2-mms-300m-az"

LLM_MODEL = "gemma-vllm"
GEMMA_SC_MODEL = "gemma4-26b-moe"

GPT_DEPLOYMENT = "gpt-chat"
GPT_API_VERSION = "2025-03-01-preview"
GPT_MODEL = "gpt-chat"
GPT_ENDPOINT = f"/openai/deployments/{GPT_DEPLOYMENT}/chat/completions?api-version={GPT_API_VERSION}"

# One of: "gemma", "gemma4" (non-thinking), "gemma4_think" (thinking mode), "gpt".
ACTIVE_LLM_PROVIDER = "gemma"

TTS_VOICE = "alloy"
TTS_ENDPOINT = "/openai/deployments/tts/audio/speech?api-version=2025-03-01-preview"

AZURE_SPEECH_BASE_URL = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com"
AZURE_SPEECH_ENDPOINT = "/cognitiveservices/v1"
AZURE_SPEECH_VOICE = "en-US-AvaMultilingualNeural"
AZURE_SPEECH_OUTPUT_FORMAT = "raw-24khz-16bit-mono-pcm"

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_MODEL = "eleven_v3"


# --- Global clients (created at import) -----------------------------------

def _client(base_url: str, *, http2: bool = False, verify: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, limits=_LIMITS, timeout=_TIMEOUT, http2=http2, verify=verify,
    )


STT_CLIENT = _client(STT_BASE_URL)
LLM_CLIENT = _client(LLM_BASE_URL)
GEMMA_SC_CLIENT = _client(GEMMA_SC_BASE_URL, verify=False)  # self-signed cert
GPT_CLIENT = _client(GPT_BASE_URL)
TTS_CLIENT = _client(TTS_BASE_URL, http2=True)
AZURE_SPEECH_CLIENT = _client(AZURE_SPEECH_BASE_URL, http2=True)
ELEVENLABS_CLIENT = _client(ELEVENLABS_BASE_URL)


# --- LLM provider routing --------------------------------------------------

def set_llm_provider(provider: str) -> None:
    """Switch LLM provider: 'gemma', 'gemma4', 'gemma4_think', or 'gpt'."""
    global ACTIVE_LLM_PROVIDER
    ACTIVE_LLM_PROVIDER = provider
    print(f"   LLM Provider set to: {provider.upper()}")


def get_active_llm_client() -> httpx.AsyncClient:
    if ACTIVE_LLM_PROVIDER in ("gemma4", "gemma4_think"):
        return GEMMA_SC_CLIENT
    if ACTIVE_LLM_PROVIDER == "gpt":
        return GPT_CLIENT
    return LLM_CLIENT


def get_active_llm_model() -> str:
    if ACTIVE_LLM_PROVIDER in ("gemma4", "gemma4_think"):
        return GEMMA_SC_MODEL
    if ACTIVE_LLM_PROVIDER == "gpt":
        return GPT_MODEL
    return LLM_MODEL


def get_active_llm_headers() -> dict:
    if ACTIVE_LLM_PROVIDER in ("gemma4", "gemma4_think"):
        return {"Content-Type": "application/json"}
    if ACTIVE_LLM_PROVIDER == "gpt":
        return {"api-key": GPT_API_KEY, "Content-Type": "application/json"}
    return {"Authorization": f"Bearer {LLM_API_KEY}"}


def get_active_llm_endpoint() -> str:
    return GPT_ENDPOINT if ACTIVE_LLM_PROVIDER == "gpt" else "/v1/chat/completions"


# --- Auth header helpers ---------------------------------------------------

def get_stt_headers() -> dict:
    return {"Authorization": f"Bearer {STT_API_KEY}"}


def get_tts_headers() -> dict:
    return {"api-key": TTS_API_KEY, "Content-Type": "application/json"}


def get_azure_speech_headers() -> dict:
    return {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": AZURE_SPEECH_OUTPUT_FORMAT,
        "User-Agent": "STSBackbone",
    }


def get_elevenlabs_headers() -> dict:
    return {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}


# --- Connection warmers ----------------------------------------------------

_warm: Dict[str, bool] = {
    "stt": False, "gemma": False, "gemma_sc": False, "gpt": False,
    "tts": False, "azure_speech": False, "elevenlabs": False,
}


async def _timed(name: str, coro_factory: Callable[[], Awaitable[httpx.Response]]) -> float:
    start = time.time()
    try:
        response = await coro_factory()
        _ = response.content
        _warm[name] = True
        return (time.time() - start) * 1000
    except Exception as e:
        print(f"   {name} warm failed: {e}")
        return 0.0


def _silent_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


async def warm_stt() -> float:
    return await _timed("stt", lambda: STT_CLIENT.post(
        "/v1/audio/transcriptions",
        data={"model": STT_MODEL},
        files={"file": ("warm.wav", _silent_wav(), "audio/wav")},
        headers=get_stt_headers(),
    ))


async def warm_gemma() -> float:
    return await _timed("gemma", lambda: LLM_CLIENT.post(
        "/v1/chat/completions",
        json={"model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
    ))


async def warm_gemma_sc() -> float:
    """Warm the secondary LLM endpoint. One warmer covers both
    --gemma4 and --gemma4-think since they share host, model, and TLS."""
    return await _timed("gemma_sc", lambda: GEMMA_SC_CLIENT.post(
        "/v1/chat/completions",
        json={"model": GEMMA_SC_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        headers={"Content-Type": "application/json"},
    ))


async def warm_gpt() -> float:
    return await _timed("gpt", lambda: GPT_CLIENT.post(
        GPT_ENDPOINT,
        json={"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 1},
        headers={"api-key": GPT_API_KEY, "Content-Type": "application/json"},
    ))


async def warm_tts() -> float:
    return await _timed("tts", lambda: TTS_CLIENT.post(
        TTS_ENDPOINT,
        json={"model": "tts-1", "input": "warm", "voice": TTS_VOICE, "response_format": "wav"},
        headers=get_tts_headers(),
    ))


async def warm_azure_speech() -> float:
    ssml = (
        f'<speak version="1.0" xml:lang="en-US">'
        f'<voice name="{AZURE_SPEECH_VOICE}">.</voice></speak>'
    )
    return await _timed("azure_speech", lambda: AZURE_SPEECH_CLIENT.post(
        AZURE_SPEECH_ENDPOINT,
        content=ssml.encode("utf-8"),
        headers=get_azure_speech_headers(),
    ))


async def ensure_azure_speech_warm() -> None:
    if not _warm["azure_speech"]:
        await warm_azure_speech()


async def warm_elevenlabs() -> float:
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY":
        print("   ElevenLabs API key not set, skipping warm")
        return 0.0
    return await _timed("elevenlabs", lambda: ELEVENLABS_CLIENT.post(
        f"/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
        json={
            "text": "warm",
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        params={"output_format": "pcm_24000"},
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
    ))


async def warm_all_connections() -> None:
    print("=" * 60)
    print("WARMING ALL CONNECTIONS")
    print("=" * 60)

    start = time.time()
    stt_ms, gemma_ms, gemma_sc_ms, gpt_ms, tts_ms, az_speech_ms, el_ms = await asyncio.gather(
        warm_stt(), warm_gemma(), warm_gemma_sc(), warm_gpt(), warm_tts(),
        warm_azure_speech(), warm_elevenlabs(),
        return_exceptions=False,
    )
    total = (time.time() - start) * 1000

    def mark(ok: bool) -> str: return "OK" if ok else "FAIL"
    print(f"   STT:            {stt_ms:.0f}ms {mark(_warm['stt'])}")
    print(f"   LLM (Primary):  {gemma_ms:.0f}ms {mark(_warm['gemma'])}")
    print(f"   LLM (Secondary):{gemma_sc_ms:.0f}ms {mark(_warm['gemma_sc'])}")
    print(f"   LLM (GPT):      {gpt_ms:.0f}ms {mark(_warm['gpt'])}")
    print(f"   TTS (REST):     {tts_ms:.0f}ms {mark(_warm['tts'])}")
    print(f"   TTS (Speech):   {az_speech_ms:.0f}ms {mark(_warm['azure_speech'])}")
    print(f"   TTS (EL):       {el_ms:.0f}ms {mark(_warm['elevenlabs'])}")
    print(f"   Total:          {total:.0f}ms")
    print("=" * 60)


async def ensure_tts_warm() -> None:
    if not _warm["tts"]:
        await warm_tts()


async def ensure_elevenlabs_warm() -> None:
    if not _warm["elevenlabs"]:
        await warm_elevenlabs()

# -*- coding: utf-8 -*-
"""
api_keys.py — Loads endpoints and secrets from the environment.

Real values live in `.env` (gitignored). Use `.env.example` as a template.
This module calls `load_dotenv()` so any importer transparently sees the
populated env vars.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _req(name: str, default: str = "") -> str:
    """Read an env var; return default if unset."""
    return os.environ.get(name, default)


# STT
STT_BASE_URL = _req("STT_BASE_URL")
STT_API_KEY = _req("STT_API_KEY")

# LLM — Provider A
LLM_BASE_URL = _req("LLM_BASE_URL")
LLM_API_KEY = _req("LLM_API_KEY")

# LLM — Provider B
GEMMA_SC_BASE_URL = _req("GEMMA_SC_BASE_URL")
GEMMA_SC_API_KEY = _req("GEMMA_SC_API_KEY")

# LLM — Provider C (OpenAI-compatible)
GPT_BASE_URL = _req("GPT_BASE_URL")
GPT_API_KEY = _req("GPT_API_KEY")

# TTS — Provider A (OpenAI-compatible)
TTS_BASE_URL = _req("TTS_BASE_URL")
TTS_API_KEY = _req("TTS_API_KEY")

# TTS — Provider B (Cognitive Services REST)
AZURE_SPEECH_REGION = _req("AZURE_SPEECH_REGION", "eastus")
AZURE_SPEECH_KEY = _req("AZURE_SPEECH_KEY")

# TTS — Provider C
ELEVENLABS_API_KEY = _req("ELEVENLABS_API_KEY")

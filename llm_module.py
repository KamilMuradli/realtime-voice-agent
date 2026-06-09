# -*- coding: utf-8 -*-
"""
llm_module.py - Streaming LLM responses with chunked dispatch for early TTS.

Streams from an OpenAI-compatible /chat/completions endpoint and invokes
on_word_chunk(chunk_text, is_first) as soon as enough words have arrived:
  * after FIRST_TRIGGER_WORDS for the first chunk (low TTFB to TTS)
  * after CHUNK_WORD_SIZE for each subsequent chunk
Avoids splitting mid-word or across known punctuation boundaries.

ChatHistory enforces strict user/assistant alternation when serialising for
the API; the conversation loop may produce out-of-order entries during
interruptions, so a single normaliser runs at send time.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

import connection_pool
from connection_pool import (
    get_active_llm_client,
    get_active_llm_endpoint,
    get_active_llm_headers,
    get_active_llm_model,
)


# --- Optional context file injection --------------------------------------

CONTEXT_FILE_PATH: Optional[str] = None


def load_context_from_file(file_path: str) -> str:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"   📄 Loaded context from: {file_path} ({len(content)} chars)")
            return content
        print(f"   ⚠️ Context file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"   ⚠️ Error loading context file: {e}")
        return ""


_loaded_context = load_context_from_file(CONTEXT_FILE_PATH) if CONTEXT_FILE_PATH else ""


# --- Metrics --------------------------------------------------------------

@dataclass
class LLMStreamMetrics:
    request_start: float = 0.0
    ttft: float = 0.0
    first_3_words_time: float = 0.0
    stream_end: float = 0.0
    total_tokens: int = 0
    full_response: str = ""

    @property
    def ttft_ms(self) -> float:
        return (self.ttft - self.request_start) * 1000 if self.ttft else 0

    @property
    def glue_latency_ms(self) -> float:
        return (self.first_3_words_time - self.ttft) * 1000 if self.first_3_words_time and self.ttft else 0

    @property
    def total_latency_ms(self) -> float:
        return (self.stream_end - self.request_start) * 1000 if self.stream_end else 0


# --- Chat history ---------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """Sən Azərbaycan dilində danışan səsli köməkçisən.
Qısa, konkret və faydalı cavablar ver.
Cavabların danışıq üçün uyğun olmalıdır - çox uzun cümlələrdən qaç.
Hər cavab 2-3 cümlədən çox olmamalıdır."""


class ChatHistory:
    def __init__(self, system_prompt: Optional[str] = None, max_history: int = 20,
                 context_file: Optional[str] = None):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        if context_file:
            self._additional_context = load_context_from_file(context_file)
        else:
            self._additional_context = _loaded_context
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def set_context_file(self, file_path: str) -> None:
        self._additional_context = load_context_from_file(file_path)

    def set_context(self, context: str) -> None:
        self._additional_context = context

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def handle_interruption(self) -> None:
        """Ensure history ends in an assistant turn so the next user message is valid."""
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.append({"role": "assistant", "content": "..."})

    def _trim(self) -> None:
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def _normalize(self) -> List[Dict[str, str]]:
        """Collapse consecutive same-role messages so user/assistant strictly alternate.

        Two user messages in a row → merged. Two assistant messages in a row →
        keep the longer one. Out of these rules, history always begins with user.
        """
        out: List[Dict[str, str]] = []
        for msg in self.messages:
            if not out:
                if msg["role"] != "user":
                    continue  # drop leading non-user noise
                out.append(dict(msg))
                continue

            if msg["role"] == out[-1]["role"]:
                if msg["role"] == "user":
                    out[-1]["content"] += " " + msg["content"]
                else:
                    if len(msg["content"]) > len(out[-1]["content"]):
                        out[-1] = dict(msg)
            else:
                out.append(dict(msg))
        return out

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        system = self.system_prompt
        if self._additional_context:
            system = (
                f"{system}\n\n"
                "Aşağıdakı məlumatlardan istifadə edərək suallara cavab ver:\n\n"
                f"---\n{self._additional_context}\n---"
            )
        return [{"role": "system", "content": system}, *self._normalize()]

    def clear(self) -> None:
        self.messages = []

    def get_last_exchange(self) -> tuple:
        user_msg, assistant_msg = None, None
        for msg in reversed(self.messages):
            if msg["role"] == "assistant" and assistant_msg is None:
                assistant_msg = msg["content"]
            elif msg["role"] == "user" and user_msg is None:
                user_msg = msg["content"]
            if user_msg and assistant_msg:
                break
        return user_msg, assistant_msg


_chat_history = ChatHistory()


def get_chat_history() -> ChatHistory:
    return _chat_history


def set_system_prompt(prompt: str) -> None:
    _chat_history.system_prompt = prompt


def set_context_file(file_path: str) -> None:
    _chat_history.set_context_file(file_path)


def set_context(context: str) -> None:
    _chat_history.set_context(context)


def clear_chat_history() -> None:
    _chat_history.clear()


# --- Streaming with smart chunking ----------------------------------------

FIRST_TRIGGER_WORDS = 2
CHUNK_WORD_SIZE = 7
SAFE_DELIMITERS = {".", ",", "!", "?", ":", ";", '"', "'", "\n"}


def _safe_word_count(text: str, total_words: int) -> int:
    """Number of words guaranteed to be complete (last char is whitespace/punct)."""
    if not text:
        return 0
    last = text[-1]
    if last.isspace() or last in SAFE_DELIMITERS:
        return total_words
    return max(0, total_words - 1)


async def stream_llm_response(
    user_input: str,
    on_word_chunk: Optional[Callable[[str, bool], Awaitable[None]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> tuple[str, LLMStreamMetrics]:
    metrics = LLMStreamMetrics()
    metrics.request_start = time.time()

    _chat_history.add_user_message(user_input)

    provider = connection_pool.ACTIVE_LLM_PROVIDER
    is_gpt = provider == "gpt"
    is_thinking = provider == "gemma4_think"

    # Thinking mode needs a much larger budget — the model emits its
    # reasoning before the final answer. Caller-supplied max_tokens still
    # wins if it's already bigger.
    if is_thinking and max_tokens < 4096:
        max_tokens = 4096

    # Azure GPT 5.3 uses different field names and ignores temperature.
    payload = {
        "model": get_active_llm_model(),
        "messages": _chat_history.get_messages_for_api(),
        "stream": True,
        ("max_completion_tokens" if is_gpt else "max_tokens"): max_tokens,
    }
    if not is_gpt:
        payload["temperature"] = temperature
    if is_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    full_response = ""
    words_dispatched = 0
    first_chunk_sent = False
    # While thinking-mode tags are streaming, buffer raw deltas separately
    # and only feed the post-`</think>` tail into full_response so the
    # word-trigger and TTS never see the model's reasoning.
    thinking_buffer = ""
    thinking_done = not is_thinking

    try:
        async with get_active_llm_client().stream(
            "POST",
            get_active_llm_endpoint(),
            json=payload,
            headers=get_active_llm_headers(),
        ) as response:
            if response.status_code != 200:
                _chat_history.add_assistant_message("(Error: Service unavailable)")
                error_text = await response.aread()
                raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")

            async for line in response.aiter_lines():
                line = line.strip() if line else ""
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices") or []
                content = choices[0].get("delta", {}).get("content", "") if choices else ""
                if not content:
                    continue

                if metrics.ttft == 0:
                    metrics.ttft = time.time()
                metrics.total_tokens += 1

                if not thinking_done:
                    thinking_buffer += content
                    end = thinking_buffer.find("</think>")
                    if end == -1:
                        continue
                    after = thinking_buffer[end + len("</think>"):]
                    thinking_buffer = ""
                    thinking_done = True
                    if not after:
                        continue
                    full_response += after
                else:
                    full_response += content

                all_words = full_response.split()
                safe_count = _safe_word_count(full_response, len(all_words))

                if not first_chunk_sent:
                    if safe_count >= FIRST_TRIGGER_WORDS:
                        chunk_text = " ".join(all_words[:FIRST_TRIGGER_WORDS])
                        if on_word_chunk:
                            await on_word_chunk(chunk_text, is_first=True)
                        words_dispatched = FIRST_TRIGGER_WORDS
                        first_chunk_sent = True
                        metrics.first_3_words_time = time.time()
                    continue

                new_words = safe_count - words_dispatched
                if new_words <= 0:
                    continue

                # Punctuation priority + orphan prevention: if we just hit a
                # delimiter, flush up to the delimiter (allowing slightly
                # larger-than-target chunks) so 1-2 words don't lag behind.
                dispatch_count = 0
                last_safe_word = all_words[safe_count - 1]
                if last_safe_word[-1] in SAFE_DELIMITERS and new_words <= CHUNK_WORD_SIZE + 3:
                    dispatch_count = new_words
                elif new_words >= CHUNK_WORD_SIZE:
                    dispatch_count = CHUNK_WORD_SIZE

                if dispatch_count:
                    chunk_text = " ".join(
                        all_words[words_dispatched:words_dispatched + dispatch_count]
                    )
                    if on_word_chunk:
                        await on_word_chunk(chunk_text, is_first=False)
                    words_dispatched += dispatch_count

        # Flush any remaining tail.
        if on_word_chunk:
            remaining = full_response.split()[words_dispatched:]
            if remaining:
                await on_word_chunk(" ".join(remaining), is_first=False)

        metrics.stream_end = time.time()
        metrics.full_response = full_response
        _chat_history.add_assistant_message(full_response)
        return full_response, metrics

    except Exception as e:
        metrics.stream_end = time.time()
        if full_response.strip():
            _chat_history.add_assistant_message(full_response)
        elif _chat_history.messages and _chat_history.messages[-1]["role"] == "user":
            _chat_history.add_assistant_message("...")
        raise Exception(f"LLM streaming failed: {e}")

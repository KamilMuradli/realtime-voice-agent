# -*- coding: utf-8 -*-
"""
================================================================================
main_client.py - Voice Assistant Main Client (Strict Audio Sequencing)
================================================================================
FIXED: Added proper turn serialization to prevent concurrent turns causing
       user/assistant alternation errors.
"""

import argparse
import asyncio
import queue
import time
import wave
import io
import os
import signal
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import pyaudio
import webrtcvad

# Import our modules
from connection_pool import warm_all_connections, set_llm_provider
from stt_module import transcribe_audio
from llm_module import (
    stream_llm_response,
    get_chat_history,
    set_system_prompt,
)
from tts_module import (
    synthesize_speech_streaming,
    merge_wav_chunks,
    prewarm_tts_connection,
    TTSResult,
    set_tts_provider,
    TTSProvider,
)
from prompt_config import get_scenario_config


# =============================================================================
# CONFIGURATION
# =============================================================================

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD settings
VAD_AGGRESSIVENESS = 3  
SILENCE_THRESHOLD_MS = 500  
MIN_SPEECH_DURATION_MS = 300 

# TTS trigger
WORD_TRIGGER_COUNT = 3 


# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-t', '--tts',
        type=str,
        choices=['azure', 'elevenlabs', 'azure-speech'],
        default='azure',
        help='TTS provider: azure (OpenAI REST), elevenlabs, azure-speech (Speech SDK - lowest latency)'
    )
    parser.add_argument('-s', '--scenario', type=str,
                        default='default',
                        help='Scenario name from prompt_config.py')
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument('--gemma', action='store_true',
                           help='Primary LLM endpoint (default)')
    llm_group.add_argument('--gemma4', action='store_true',
                           help='Secondary LLM endpoint (no thinking)')
    llm_group.add_argument('--gemma4-think', dest='gemma4_think', action='store_true',
                           help='Secondary LLM endpoint with thinking mode')
    llm_group.add_argument('--gpt', action='store_true',
                           help='GPT-compatible LLM endpoint')
    parser.add_argument(
        '--no-interrupt',
        action='store_true',
        default=False,
        help='Disable barge-in: Mute VAD while bot is speaking to prevent self-interruption (no headphones needed)'
    )
    parser.add_argument(
        '--allow-interrupt',
        action='store_true',
        default=True,
        help='Enable barge-in: Allow user to interrupt the bot (requires headphones to avoid self-interruption)'
    )
    return parser.parse_args()


# =============================================================================
# LATENCY TRACKER
# =============================================================================

@dataclass
class TurnLatency:
    vad_end: float = 0.0
    stt_send: float = 0.0
    stt_complete: float = 0.0
    prewarm_complete: float = 0.0
    llm_start: float = 0.0
    llm_ttft: float = 0.0
    llm_3_words: float = 0.0
    llm_stream_end: float = 0.0
    tts_start: float = 0.0
    tts_ttfb: float = 0.0
    audio_play_start: float = 0.0
    turn_complete: float = 0.0
    speech_duration_ms: float = 0.0
    transcription: str = ""
    response: str = ""
    tts_provider: str = ""
    tts_chunk_metrics: list = field(default_factory=list)  # [(chunk_id, ttfb_ms, total_ms, bytes)]

    def get_ear_to_mouth(self) -> float:
        return (self.tts_ttfb - self.vad_end) * 1000 if self.tts_ttfb else 0

    def _ms(self, t_start: float, t_end: float) -> str:
        if t_start and t_end and t_end > t_start:
            return f"{(t_end - t_start) * 1000:>8.1f} ms"
        return "     --- ms"

    def print_report(self):
        W = 70
        sep  = "=" * W
        thin = "-" * W

        print(f"\n{sep}")
        print(f"  LATENCY REPORT  [TTS: {self.tts_provider.upper()}]")
        print(sep)
        print(f'  Input : "{self.transcription}"  ({self.speech_duration_ms:.0f} ms audio)')
        resp_preview = self.response[:80] + ("..." if len(self.response) > 80 else "")
        print(f'  Output: "{resp_preview}"')
        print(thin)

        # --- STT ---
        print("  STT")
        print(f"    Queue delay   (VAD end → send)     :{self._ms(self.vad_end,    self.stt_send)}")
        print(f"    Inference     (send → complete)    :{self._ms(self.stt_send,   self.stt_complete)}")
        print(f"    Total STT     (VAD end → complete) :{self._ms(self.vad_end,    self.stt_complete)}")

        # --- PREWARM (runs concurrently with STT) ---
        if self.prewarm_complete:
            print("  PREWARM  (concurrent with STT)")
            print(f"    TTS pre-warm  (send → ready)       :{self._ms(self.stt_send, self.prewarm_complete)}")

        # --- LLM ---
        llm_total = self._ms(self.llm_start, self.llm_stream_end) if self.llm_stream_end else "     --- ms"
        print("  LLM")
        print(f"    Start delay   (STT done → send)    :{self._ms(self.stt_complete, self.llm_start)}")
        print(f"    TTFT          (send → 1st token)   :{self._ms(self.llm_start,   self.llm_ttft)}")
        print(f"    Glue          (1st token → 3 wds)  :{self._ms(self.llm_ttft,    self.llm_3_words)}")
        print(f"    Stream total  (send → stream end)  : {llm_total}")

        # --- TTS chunks ---
        if self.tts_chunk_metrics:
            for cid, ttfb_ms, total_ms, nbytes in sorted(self.tts_chunk_metrics):
                print(f"  TTS Chunk #{cid}")
                print(f"    TTFB          (send → 1st byte)   :  {ttfb_ms:>8.1f} ms")
                print(f"    Total         (send → complete)   :  {total_ms:>8.1f} ms")
                print(f"    Audio size                        :  {nbytes / 1024:>8.1f} KB")

        # --- Pipeline totals ---
        e2m = self.get_ear_to_mouth()
        turn_total_ms = (self.turn_complete - self.vad_end) * 1000 if self.turn_complete else 0
        print(thin)
        print("  PIPELINE TOTALS")
        print(f"    Ear-to-Mouth  (VAD end → 1st audio) : {e2m:>8.1f} ms")
        if turn_total_ms:
            print(f"    Turn complete (VAD end → all done)  : {turn_total_ms:>8.1f} ms")
        print(sep + "\n")


# =============================================================================
# AUDIO PLAYBACK QUEUE (The Sequencer)
# =============================================================================

class AudioPlaybackQueue:
    def __init__(self, on_playback_start=None, on_playback_end=None):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.on_playback_start = on_playback_start
        self.on_playback_end = on_playback_end
        self._is_playing = False
        self._sample_rate = 24000
        self._channels = 1
        self._sample_width = 2
        
        # SEQUENCING VARIABLES
        self._current_generation = 0
        self._next_expected_chunk_id = 0
        self._chunk_buffer: Dict[int, bytes] = {} # "Waiting Room" for out-of-order chunks
    
    @property
    def is_playing(self) -> bool:
        return self._is_playing
    
    def start(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self._running = True
        self._task = asyncio.create_task(self._playback_loop())
        print("🔊 Audio playback queue started")
    
    async def stop(self):
        self._running = False
        await self.queue.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        print("🔇 Audio playback queue stopped")

    async def wait_until_empty(self):
        """Wait until all queued audio has finished playing"""
        while not self.queue.empty() or self._is_playing or self._chunk_buffer:
            await asyncio.sleep(0.1)

    async def clear(self):
        """Interrupt logic: Clear queue AND sequencing buffer"""
        self._current_generation += 1
        
        # Reset sequencer for the new turn
        self._next_expected_chunk_id = 0
        self._chunk_buffer.clear()
        
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._is_playing = False
        print("   🔇 Playback cleared (Interrupted - Safe Mode)")

    async def add_audio(self, audio_data: bytes, chunk_id: int = 0):
        # We put everything in the queue, even empty bytes, to keep the ID sequence intact
        await self.queue.put((audio_data, chunk_id, self._current_generation))
    
    async def signal_end(self):
        await self.queue.put(("END_TURN", -1, self._current_generation))
    
    async def _playback_loop(self):
        while self._running:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                if item is None:
                    break
                
                msg_type_or_data, chunk_id, generation = item
                
                # 1. Discard audio from previous (interrupted) turns
                if generation != self._current_generation:
                    continue
                
                # 2. Handle End of Turn
                if msg_type_or_data == "END_TURN":
                    self._next_expected_chunk_id = 0 # Reset for safety
                    self._chunk_buffer.clear()
                    
                    if self._is_playing:
                        self._is_playing = False
                        if self.on_playback_end:
                            self.on_playback_end()
                    continue
                
                # 3. SEQUENCING LOGIC
                # Buffer the incoming chunk
                audio_data = msg_type_or_data
                self._chunk_buffer[chunk_id] = audio_data
                
                # 4. Play from buffer ONLY if it matches _next_expected_chunk_id
                while self._next_expected_chunk_id in self._chunk_buffer:
                    
                    # Pop the correct next chunk
                    next_audio = self._chunk_buffer.pop(self._next_expected_chunk_id)
                    self._next_expected_chunk_id += 1
                    
                    # If audio bytes are empty (e.g. [HANGUP] tag removed), skip playing but count it as done
                    if not next_audio:
                        continue

                    if not self._is_playing:
                        self._is_playing = True
                        if self.on_playback_start:
                            self.on_playback_start()
                    
                    # Decode and Play
                    try:
                        wav_io = io.BytesIO(next_audio)
                        with wave.open(wav_io, 'rb') as wav:
                            self._sample_rate = wav.getframerate()
                            self._channels = wav.getnchannels()
                            self._sample_width = wav.getsampwidth()
                            frames = wav.readframes(wav.getnframes())
                    except Exception:
                        frames = next_audio
                        self._sample_rate = 24000
                        self._channels = 1
                        self._sample_width = 2

                    new_rate = self._sample_rate
                    if self.stream is None or getattr(self.stream, '_rate', None) != new_rate:
                        if self.stream:
                            self.stream.stop_stream()
                            self.stream.close()
                        self.stream = self.pyaudio_instance.open(
                            format=self.pyaudio_instance.get_format_from_width(self._sample_width),
                            channels=self._channels,
                            rate=new_rate,
                            output=True
                        )
                        self.stream._rate = new_rate
                    
                    # Re-check generation before blocking write (in case of interrupt during loop)
                    if generation != self._current_generation:
                        break
                        
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.stream.write, frames)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ Playback error: {e}")
                continue


# =============================================================================
# VAD AUDIO CAPTURE
# =============================================================================

class VADAudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(aggressiveness)
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.speech_queue: queue.Queue = queue.Queue()
        self._running = False
        self._speech_buffer: List[bytes] = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._silence_frame_threshold = int(SILENCE_THRESHOLD_MS / CHUNK_DURATION_MS)
        self._min_speech_frames = int(MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS)

        # Mute control - when True, VAD ignores all input (prevents self-interruption)
        self._muted = False
    
    def start(self):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._audio_callback
        )
        self._running = True
        self.stream.start_stream()
        print("🎤 Microphone capture started")
    
    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        self.speech_queue.put(None)
        print("🎤 Microphone capture stopped")

    def mute(self):
        """Mute VAD - ignores all audio input (used during bot playback to prevent self-interruption)"""
        self._muted = True
        # Clear any partial speech buffer when muting
        self._speech_buffer = []
        self._speech_frames = 0
        self._silence_frames = 0

    def unmute(self):
        """Unmute VAD - resumes listening for speech"""
        self._muted = False

    @property
    def is_muted(self) -> bool:
        return self._muted

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if not self._running:
            return (None, pyaudio.paComplete)

        # Skip processing if muted (prevents self-interruption during bot playback)
        if self._muted:
            return (None, pyaudio.paContinue)

        try:
            is_speech = self.vad.is_speech(in_data, self.sample_rate)
            if is_speech:
                self._speech_buffer.append(in_data)
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                if self._speech_frames > 0:
                    self._speech_buffer.append(in_data)
                    self._silence_frames += 1
                    if self._silence_frames >= self._silence_frame_threshold:
                        if self._speech_frames >= self._min_speech_frames:
                            audio_data = b''.join(self._speech_buffer)
                            timestamp = time.time()
                            try:
                                self.speech_queue.put_nowait((audio_data, timestamp))
                            except queue.Full:
                                pass
                        self._speech_buffer = []
                        self._speech_frames = 0
                        self._silence_frames = 0
        except Exception:
            pass
        return (None, pyaudio.paContinue)
    
    async def get_speech(self) -> Optional[tuple]:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.speech_queue.get(timeout=0.1)
                )
                return result
            except queue.Empty:
                continue
            except Exception:
                continue
        return None


# =============================================================================
# VOICE ASSISTANT (With Logging, Barge-In, & Scenarios)
# =============================================================================

class VoiceAssistant:
    def __init__(self, tts_provider: TTSProvider = TTSProvider.AZURE, scenario: str = "default", allow_interrupt: bool = True):
        self.audio_capture: Optional[VADAudioCapture] = None
        self.playback_queue: Optional[AudioPlaybackQueue] = None
        self._running = False
        self._turn_count = 0
        self._tts_provider = tts_provider
        self._scenario = scenario
        self._config = get_scenario_config(scenario)  # Load specific config
        self._processing_task: Optional[asyncio.Task] = None

        # INTERRUPTION MODE
        # allow_interrupt=True: User can interrupt bot (requires headphones)
        # allow_interrupt=False: VAD muted during playback (no headphones needed)
        self._allow_interrupt = allow_interrupt
        
        # FIX: Lock to serialize turn processing in no-interrupt mode
        self._turn_lock = asyncio.Lock()

        # LOGGING SETUP
        self._log_dir = os.path.join("logs", f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(self._log_dir, exist_ok=True)
        print(f"   📂 Logging enabled: {self._log_dir}")
    
    async def initialize(self):
        print("\n" + "="*70)
        print("🤖 VOICE ASSISTANT INITIALIZING")
        print("="*70)

        # Set System Prompt
        set_system_prompt(self._config["prompt"])
        print(f"   🎭 Scenario: {self._scenario.upper()}")
        print(f"   🌡️ LLM Temp: {self._config['temp']}")

        set_tts_provider(self._tts_provider)
        print(f"   🔊 TTS Provider: {self._tts_provider.value.upper()}")

        # Interruption mode
        if self._allow_interrupt:
            print(f"   🎧 Interruption: ENABLED (barge-in allowed, use headphones!)")
        else:
            print(f"   🔇 Interruption: DISABLED (VAD muted during playback, no headphones needed)")

        await warm_all_connections()

        self.audio_capture = VADAudioCapture()

        # Set up playback callbacks for VAD muting (only when interruption is disabled)
        if self._allow_interrupt:
            # Allow interruption - no muting
            self.playback_queue = AudioPlaybackQueue()
        else:
            # Disable interruption - mute VAD during playback
            def on_playback_start():
                if self.audio_capture:
                    self.audio_capture.mute()
                    print("   🔇 VAD muted (bot speaking)")

            def on_playback_end():
                if self.audio_capture:
                    self.audio_capture.unmute()
                    print("   🎤 VAD unmuted (listening)")

            self.playback_queue = AudioPlaybackQueue(
                on_playback_start=on_playback_start,
                on_playback_end=on_playback_end
            )

        self.playback_queue.start()

        print("\n✅ Voice Assistant ready!")
    
    async def start(self):
        self._running = True
        
        # --- BOT-FIRST INITIATION ---
        # If the scenario has an opening line, speak it BEFORE listening
        if self._config.get("first_turn"):
            opening_text = self._config["first_turn"]
            print(f"\n📢 Bot Starting: \"{opening_text}\"")
            
            # 1. Synthesize immediately (Chunk 0)
            await self._synthesize_and_queue(opening_text, 0, None)
            await self.playback_queue.signal_end() # Signal end of bot turn
            
            # 2. Inject into history so LLM knows it started
            get_chat_history().add_user_message("(Zəng başladı)")
            get_chat_history().add_assistant_message(opening_text)
            
        print("   Speak into your microphone. (Headphones recommended for Barge-In)\n")
        self.audio_capture.start()
        
        try:
            while self._running:
                result = await self.audio_capture.get_speech()
                if result is None:
                    break
                audio_data, vad_end_time = result

                # --- BARGE-IN INTERRUPTION CHECK (only if interruption is enabled) ---
                if self._allow_interrupt:
                    is_speaking = self.playback_queue.is_playing
                    is_thinking = (self._processing_task is not None and not self._processing_task.done())

                    if is_speaking or is_thinking:
                        print(f"\n👂 Interruption detected! (Speaking: {is_speaking}, Thinking: {is_thinking})")
                        await self.playback_queue.clear()
                        if self._processing_task:
                            self._processing_task.cancel()
                            try:
                                await self._processing_task
                            except asyncio.CancelledError:
                                print("   🛑 Previous turn cancelled successfully")
                            except Exception as e:
                                print(f"   ⚠️ Previous turn error during cancel: {e}")
                        
                        # FIX: Call handle_interruption AFTER awaiting the cancelled task
                        # This ensures the task has fully stopped before we fix history
                        get_chat_history().handle_interruption()
                        
                        # Small yield to ensure everything is settled
                        await asyncio.sleep(0.01)

                    # --- PROCESS NEW INPUT (with interruption) ---
                    self._processing_task = asyncio.create_task(
                        self._handle_speech_end(audio_data, vad_end_time)
                    )
                else:
                    # --- NO-INTERRUPT MODE: SERIALIZE TURNS ---
                    # FIX: Wait for any previous turn to complete before starting new one
                    # This prevents the 400 Bad Request error from concurrent user messages
                    
                    # Check if there's already a turn being processed
                    if self._processing_task is not None and not self._processing_task.done():
                        print(f"\n⏳ Previous turn still processing, queuing new input...")
                        # Wait for the previous task to complete
                        try:
                            await self._processing_task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            print(f"   ⚠️ Previous turn had error: {e}")
                    
                    # Now safe to process the new turn
                    self._processing_task = asyncio.create_task(
                        self._handle_speech_end(audio_data, vad_end_time)
                    )
                    
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        self._running = False
        if self.audio_capture:
            self.audio_capture.stop()
        if self.playback_queue:
            await self.playback_queue.stop()
        print("\n👋 Voice Assistant stopped")
    
    async def _handle_speech_end(self, audio_data: bytes, vad_end_time: float):
        try:
            self._turn_count += 1
            turn = TurnLatency()
            turn.vad_end = vad_end_time
            turn.speech_duration_ms = len(audio_data) / (SAMPLE_RATE * SAMPLE_WIDTH) * 1000
            turn.tts_provider = self._tts_provider.value
            
            print(f"\n{'─'*70}")
            print(f"🎤 Turn {self._turn_count}: Speech detected ({turn.speech_duration_ms:.0f}ms)")
            
            prewarm_task = asyncio.create_task(prewarm_tts_connection())
            
            turn.stt_send = time.time()
            stt_result = await transcribe_audio(audio_data, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH)
            turn.stt_complete = time.time()
            
            if not stt_result.success or not stt_result.text:
                print(f"   ⚠️ STT failed: {stt_result.error or 'No transcription'}")
                return
            
            turn.transcription = stt_result.text
            print(f"   📝 \"{turn.transcription}\" ({stt_result.latency_ms:.0f}ms)")
            
            await prewarm_task
            turn.prewarm_complete = time.time()

            turn.llm_start = time.time()
            print("   🤖 Generating response...")
            
            tts_tasks = []
            chunk_id_counter = 0
            
            async def dispatch_tts(text: str, is_first: bool):
                nonlocal chunk_id_counter
                if is_first:
                    turn.llm_3_words = time.time()
                    turn.tts_start = time.time()
                    print(f"   🎙️ TTS[0]: \"{text}\"")
                
                turn_ref = turn if is_first else None
                task = asyncio.create_task(
                    self._synthesize_and_queue(text, chunk_id_counter, turn_ref)
                )
                tts_tasks.append(task)
                chunk_id_counter += 1
            
            # USE SPECIFIC TEMPERATURE FROM CONFIG
            full_response, llm_metrics = await stream_llm_response(
                turn.transcription,
                on_word_chunk=dispatch_tts,
                temperature=self._config["temp"]
            )
            
            turn.llm_ttft = llm_metrics.ttft
            turn.llm_stream_end = llm_metrics.stream_end
            turn.response = full_response

            tts_results = []
            if tts_tasks:
                tts_results = await asyncio.gather(*tts_tasks, return_exceptions=True)

            # Collect per-chunk TTS metrics
            for res in tts_results:
                if isinstance(res, tuple) and len(res) == 3 and res[2] is not None:
                    chunk_id, _, tts_result = res
                    turn.tts_chunk_metrics.append((
                        chunk_id,
                        tts_result.ttfb_ms,
                        tts_result.total_latency_ms,
                        tts_result.audio_bytes
                    ))

            await self.playback_queue.signal_end()
            turn.turn_complete = time.time()
            turn.print_report()
            
            self._save_turn_logs(self._turn_count, full_response, tts_results)
            
            # --- AUTO TERMINATION CHECK ---
            if "[HANGUP]" in full_response:
                print("\n📵 HANGUP SIGNAL DETECTED. Ending call after playback.")
                await self.playback_queue.wait_until_empty()
                self._running = False
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"   ❌ Error in turn processing: {e}")
            import traceback
            traceback.print_exc()
    
    async def _synthesize_and_queue(self, text: str, chunk_id: int, turn: Optional[TurnLatency]) -> Tuple[int, Optional[bytes], Optional[TTSResult]]:
        # Strip [HANGUP] before sending to TTS
        clean_text = text.replace("[HANGUP]", "").strip()

        # CRITICAL FIX: Always send signal to queue, even if text is empty
        if not clean_text:
            await self.playback_queue.add_audio(b'', chunk_id)
            return (chunk_id, None, None)

        try:
            result = await synthesize_speech_streaming(clean_text)
            if result.success:
                if turn is not None and chunk_id == 0:
                    turn.tts_ttfb = time.time()
                await self.playback_queue.add_audio(result.audio_data, chunk_id)
                return (chunk_id, result.audio_data, result)
            else:
                # Send empty bytes on failure to keep sequence moving
                await self.playback_queue.add_audio(b'', chunk_id)
                return (chunk_id, None, None)
        except Exception as e:
            print(f"   ⚠️ TTS error: {e}")
            await self.playback_queue.add_audio(b'', chunk_id)
            return (chunk_id, None, None)

    def _save_turn_logs(self, turn_id: int, text: str, tts_results: List):
        try:
            text_filename = os.path.join(self._log_dir, f"turn_{turn_id:03d}.txt")
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text)
            
            valid_chunks = [res for res in tts_results if isinstance(res, tuple) and res[1] is not None]
            
            if valid_chunks:
                valid_chunks.sort(key=lambda x: x[0])
                audio_bytes_list = [x[1] for x in valid_chunks]
                full_audio = merge_wav_chunks(audio_bytes_list)
                wav_filename = os.path.join(self._log_dir, f"turn_{turn_id:03d}.wav")
                with open(wav_filename, 'wb') as f:
                    f.write(full_audio)
                print(f"   💾 Logged: {text_filename} | {wav_filename}")
        except Exception as e:
            print(f"   ⚠️ Logging failed: {e}")


# =============================================================================
# MAIN (Fixed Race Condition)
# =============================================================================

async def main(tts_provider: TTSProvider, scenario: str, allow_interrupt: bool = True):
    assistant = VoiceAssistant(tts_provider=tts_provider, scenario=scenario, allow_interrupt=allow_interrupt)
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        print("\n\n⚠️ Interrupt received, shutting down...")
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await assistant.initialize()
        
        # Start the assistant as a task
        assistant_task = asyncio.create_task(assistant.start())
        stop_task = asyncio.create_task(stop_event.wait())
        
        # Wait for the task to finish (happens if _running becomes False) OR user interrupt
        done, pending = await asyncio.wait(
            [assistant_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        await assistant.stop()


if __name__ == "__main__":
    args = parse_args()

    # Set LLM provider
    if args.gemma4_think:
        set_llm_provider("gemma4_think")
    elif args.gemma4:
        set_llm_provider("gemma4")
    elif args.gpt:
        set_llm_provider("gpt")
    else:
        set_llm_provider("gemma")

    # Map command line arg to TTSProvider enum
    if args.tts == 'elevenlabs':
        tts_provider = TTSProvider.ELEVENLABS
    elif args.tts == 'azure-speech':
        tts_provider = TTSProvider.AZURE_SPEECH
    else:
        tts_provider = TTSProvider.AZURE

    # Determine interruption mode
    # --no-interrupt takes precedence if specified
    allow_interrupt = not args.no_interrupt

    if allow_interrupt:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║   🎤 HIGH-PERFORMANCE VOICE ASSISTANT (MULTI-SCENARIO)               ║
║   🎧 INTERRUPTION MODE: ENABLED (Barge-in allowed)                   ║
║   ⚠️  USE HEADPHONES TO AVOID SELF-INTERRUPTION                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║   🎤 HIGH-PERFORMANCE VOICE ASSISTANT (MULTI-SCENARIO)               ║
║   🔇 INTERRUPTION MODE: DISABLED (VAD muted during playback)         ║
║   ✅ NO HEADPHONES NEEDED - Bot won't interrupt itself               ║
╚══════════════════════════════════════════════════════════════════════╝
        """)

    asyncio.run(main(tts_provider, args.scenario, allow_interrupt))
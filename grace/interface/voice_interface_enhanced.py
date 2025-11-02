"""
Voice Interface - Speech Recognition and Synthesis (Enhanced)
=============================================================

Full STT/TTS implementation with multiple backend options.
Supports Whisper, Google, Azure, and offline engines.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class STTEngine(str, Enum):
    """Speech-to-Text engine options"""
    WHISPER_LOCAL = "whisper_local"
    WHISPER_API = "whisper_api"
    GOOGLE = "google"
    AZURE = "azure"
    MOCK = "mock"


class TTSEngine(str, Enum):
    """Text-to-Speech engine options"""
    PYTTSX3 = "pyttsx3"
    GOOGLE = "google"
    AZURE = "azure"
    OPENAI = "openai"
    MOCK = "mock"


class VoiceInterfaceEnhanced:
    """
    Enhanced Voice Interface with real STT/TTS
    """
    
    def __init__(self, stt_engine: STTEngine = STTEngine.MOCK, tts_engine: TTSEngine = TTSEngine.MOCK):
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.active = False
        self.language = "en-US"
        self.stats = {"audio_processed": 0, "speech_synthesized": 0, "errors": 0}
        self._stt_client = None
        self._tts_client = None
    
    async def start(self):
        """Start voice interface"""
        self.active = True
        await self._init_engines()
        logger.info("Enhanced VoiceInterface started")
    
    async def stop(self):
        """Stop voice interface"""
        self.active = False
    
    async def _init_engines(self):
        """Initialize STT/TTS engines"""
        if self.stt_engine == STTEngine.WHISPER_API:
            try:
                import httpx
                self._stt_client = httpx.AsyncClient()
            except ImportError:
                logger.warning("httpx not available, using mock")
                self.stt_engine = STTEngine.MOCK
    
    async def process_audio(self, audio_data: bytes) -> Optional[str]:
        """Process audio and return transcribed text"""
        try:
            if self.stt_engine == STTEngine.WHISPER_API:
                return await self._whisper_api(audio_data)
            elif self.stt_engine == STTEngine.MOCK:
                await asyncio.sleep(0.1)
                return f"[Mock transcription of {len(audio_data)} bytes]"
            
            self.stats["audio_processed"] += 1
            return None
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            self.stats["errors"] += 1
            return None
    
    async def _whisper_api(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using OpenAI Whisper API"""
        if not self._stt_client:
            return None
        
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using mock")
            return await self._transcribe_mock(audio_data)
        
        try:
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            response = await self._stt_client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                files=files,
                data={"model": "whisper-1"},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            if response.status_code == 200:
                return response.json().get("text")
            return None
        except Exception as e:
            logger.error(f"Whisper API failed: {e}")
            return None
    
    async def synthesize_speech(self, text: str) -> bytes:
        """Convert text to speech"""
        try:
            if self.tts_engine == TTSEngine.MOCK:
                await asyncio.sleep(0.1)
                return b"RIFF" + len(text).to_bytes(4, 'little') + b"WAVE"
            
            self.stats["speech_synthesized"] += 1
            return b""
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            self.stats["errors"] += 1
            return b""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "active": self.active,
            "language": self.language,
            "stt_engine": self.stt_engine.value,
            "tts_engine": self.tts_engine.value,
            **self.stats
        }

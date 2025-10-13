"""
Voice Interface Service for Grace - Bidirectional Voice Communication
====================================================================

Provides voice toggle functionality for full conversation between user and Grace:
- Real-time voice recording and transcription
- Text-to-speech response generation
- Bidirectional conversation mode
- Voice activity detection
- Integration with existing Grace communication systems

Usage:
    from grace.interface_kernel.voice_service import VoiceService

    voice = VoiceService()
    await voice.start()

    # Toggle voice mode on/off
    await voice.toggle_voice_mode()
"""

import asyncio
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    import speech_recognition as sr
    import pyttsx3

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class VoiceState(Enum):
    """Voice interface states."""

    INACTIVE = "inactive"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class VoiceMessage:
    """Voice message container."""

    content: str
    timestamp: datetime
    source: str  # 'user' or 'grace'
    confidence: Optional[float] = None
    language: str = "en"


class VoiceService:
    """Bidirectional voice communication service for Grace."""

    def __init__(self, grace_comm_handler: Optional[Callable] = None):
        self.state = VoiceState.INACTIVE
        self.is_voice_enabled = False
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.grace_comm_handler = grace_comm_handler
        self.conversation_history: List[VoiceMessage] = []
        self.listening_thread = None
        self._shutdown_event = asyncio.Event()

        # Configuration
        self.config = {
            "listen_timeout": 5.0,
            "phrase_timeout": 1.0,
            "speech_rate": 180,
            "speech_volume": 0.8,
            "voice_index": 1,  # Female voice if available
            "energy_threshold": 4000,
            "pause_threshold": 0.8,
        }

        self._initialize_voice_components()

    def _initialize_voice_components(self) -> bool:
        """Initialize speech recognition and text-to-speech components."""
        if not VOICE_AVAILABLE:
            logger.warning(
                "Voice dependencies not available. Install speech_recognition and pyttsx3."
            )
            return False

        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.config["energy_threshold"]
            self.recognizer.pause_threshold = self.config["pause_threshold"]

            # Initialize microphone
            self.microphone = sr.Microphone()

            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.config["speech_rate"])
            self.tts_engine.setProperty("volume", self.config["speech_volume"])

            # Set voice (prefer female voice for Grace)
            voices = self.tts_engine.getProperty("voices")
            if voices and len(voices) > self.config["voice_index"]:
                self.tts_engine.setProperty(
                    "voice", voices[self.config["voice_index"]].id
                )

            # Calibrate microphone
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info("Voice components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            return False

    async def toggle_voice_mode(self) -> Dict[str, Any]:
        """Toggle voice communication mode on/off."""
        if not VOICE_AVAILABLE:
            return {
                "success": False,
                "message": "Voice functionality not available - missing dependencies",
                "state": self.state.value,
            }

        if self.is_voice_enabled:
            await self._stop_voice_mode()
            return {
                "success": True,
                "message": "Voice mode disabled",
                "state": self.state.value,
            }
        else:
            await self._start_voice_mode()
            return {
                "success": True,
                "message": "Voice mode enabled - listening for commands",
                "state": self.state.value,
            }

    async def _start_voice_mode(self):
        """Start voice communication mode."""
        self.is_voice_enabled = True
        self.state = VoiceState.LISTENING
        self._shutdown_event.clear()

        # Start listening thread
        self.listening_thread = threading.Thread(
            target=self._listen_continuously, daemon=True
        )
        self.listening_thread.start()

        # Announce activation
        await self._speak("Voice mode activated. I'm listening for your questions.")

        logger.info("Voice communication mode started")

    async def _stop_voice_mode(self):
        """Stop voice communication mode."""
        self.is_voice_enabled = False
        self.state = VoiceState.INACTIVE
        self._shutdown_event.set()

        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2.0)

        # Announce deactivation
        await self._speak("Voice mode deactivated.")

        logger.info("Voice communication mode stopped")

    def _listen_continuously(self):
        """Continuously listen for voice input in background thread."""
        while self.is_voice_enabled and not self._shutdown_event.is_set():
            try:
                # Listen for audio input
                with self.microphone as source:
                    logger.debug("Listening for voice input...")
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.config["listen_timeout"],
                        phrase_time_limit=None,
                    )

                # Transcribe audio
                self.state = VoiceState.PROCESSING
                text = self.recognizer.recognize_google(audio, language="en-US")

                if text.strip():
                    # Add to conversation history
                    message = VoiceMessage(
                        content=text,
                        timestamp=datetime.now(),
                        source="user",
                        confidence=None,  # Google doesn't provide confidence scores
                    )
                    self.conversation_history.append(message)

                    logger.info(f"Voice input received: {text}")

                    # Process with Grace
                    asyncio.create_task(self._handle_voice_input(text))

            except sr.WaitTimeoutError:
                # Normal timeout - continue listening
                self.state = VoiceState.LISTENING
                continue

            except sr.UnknownValueError:
                # Could not understand audio
                logger.debug("Could not understand audio")
                self.state = VoiceState.LISTENING
                continue

            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                self.state = VoiceState.LISTENING
                time.sleep(1)  # Wait before retrying

            except Exception as e:
                logger.error(f"Unexpected error in voice listening: {e}")
                self.state = VoiceState.LISTENING
                time.sleep(1)

    async def _handle_voice_input(self, text: str):
        """Handle voice input by processing with Grace and responding."""
        try:
            # Check for voice control commands
            if await self._handle_voice_commands(text):
                return

            # Process with Grace communication handler
            if self.grace_comm_handler:
                response = await self.grace_comm_handler(text)

                # Extract response text
                response_text = self._extract_response_text(response)

                # Add Grace's response to history
                grace_message = VoiceMessage(
                    content=response_text, timestamp=datetime.now(), source="grace"
                )
                self.conversation_history.append(grace_message)

                # Speak the response
                await self._speak(response_text)

            else:
                # Fallback response
                response_text = f"I heard you say: {text}. However, no communication handler is available."
                await self._speak(response_text)

        except Exception as e:
            logger.error(f"Error handling voice input: {e}")
            await self._speak(
                "I'm sorry, I encountered an error processing your request."
            )

        finally:
            self.state = VoiceState.LISTENING

    async def _handle_voice_commands(self, text: str) -> bool:
        """Handle special voice commands. Returns True if command was handled."""
        text_lower = text.lower()

        if (
            "stop voice" in text_lower
            or "disable voice" in text_lower
            or "voice off" in text_lower
        ):
            await self._stop_voice_mode()
            return True

        if "clear history" in text_lower or "clear conversation" in text_lower:
            self.conversation_history.clear()
            await self._speak("Conversation history cleared.")
            return True

        if "voice status" in text_lower or "voice state" in text_lower:
            status = f"Voice mode is active. Current state: {self.state.value}. {len(self.conversation_history)} messages in history."
            await self._speak(status)
            return True

        return False

    def _extract_response_text(self, response: Any) -> str:
        """Extract readable text from Grace's response."""
        if isinstance(response, dict):
            if "answer" in response:
                return response["answer"]
            elif "message" in response:
                return response["message"]
            elif "content" in response:
                return response["content"]
            else:
                return str(response)
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    async def _speak(self, text: str):
        """Convert text to speech and play it."""
        if not self.tts_engine:
            logger.warning("TTS engine not available")
            return

        try:
            self.state = VoiceState.SPEAKING

            # Run TTS in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._tts_speak, text)

            logger.debug(f"Spoke: {text}")

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
        finally:
            if self.is_voice_enabled:
                self.state = VoiceState.LISTENING

    def _tts_speak(self, text: str):
        """Synchronous TTS speaking (runs in thread)."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as JSON-serializable format."""
        return [
            {
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "source": msg.source,
                "confidence": msg.confidence,
                "language": msg.language,
            }
            for msg in self.conversation_history
        ]

    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice interface status."""
        return {
            "voice_enabled": self.is_voice_enabled,
            "state": self.state.value,
            "voice_available": VOICE_AVAILABLE,
            "conversation_length": len(self.conversation_history),
            "config": self.config,
        }

    async def cleanup(self):
        """Cleanup voice service resources."""
        if self.is_voice_enabled:
            await self._stop_voice_mode()

        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception:
                pass

        logger.info("Voice service cleaned up")


# Mock implementation for when voice libraries aren't available
class MockVoiceService:
    """Mock voice service when dependencies aren't available."""

    def __init__(self, grace_comm_handler: Optional[Callable] = None):
        self.is_voice_enabled = False
        self.state = VoiceState.INACTIVE
        self.conversation_history = []
        self.grace_comm_handler = grace_comm_handler

    async def toggle_voice_mode(self) -> Dict[str, Any]:
        return {
            "success": False,
            "message": "Voice functionality not available - install speech_recognition and pyttsx3",
            "state": self.state.value,
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return []

    def get_voice_status(self) -> Dict[str, Any]:
        return {
            "voice_enabled": False,
            "state": self.state.value,
            "voice_available": False,
            "conversation_length": 0,
            "config": {},
        }

    async def cleanup(self):
        pass


def create_voice_service(grace_comm_handler: Optional[Callable] = None):
    """Factory function to create appropriate voice service."""
    if VOICE_AVAILABLE:
        return VoiceService(grace_comm_handler)
    else:
        return MockVoiceService(grace_comm_handler)

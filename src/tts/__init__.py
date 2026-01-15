"""Text-to-speech module."""
from .bark_tts import generate_speech
from .replicate_tts import generate_speech_replicate

__all__ = ["generate_speech", "generate_speech_replicate"]

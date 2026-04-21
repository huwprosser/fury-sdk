from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Optional

from .utils.voice import DEFAULT_TRANSCRIPTION_MODEL_NAME, _create_transcription_model
from .utils.voice import transcribe_audio as _transcribe_audio


class SpeechToText:
    """Reusable Faster Whisper speech-to-text helper."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_TRANSCRIPTION_MODEL_NAME,
        prewarm: bool = True,
        suppress_logs: bool = False,
    ) -> None:
        self.model_name = model_name
        self.suppress_logs = suppress_logs
        self.model: Optional[Any] = None

        if prewarm:
            self.prewarm()

    def prewarm(self) -> Any:
        if self.model is None:
            if not self.suppress_logs:
                print("Warming up STT...")
            self.model = _create_transcription_model(self.model_name)
        return self.model

    def transcribe(self, audio: str | bytes | Path | BinaryIO) -> str:
        return _transcribe_audio(audio, model=self.prewarm())

    def __call__(self, audio: str | bytes | Path | BinaryIO) -> str:
        return self.transcribe(audio)

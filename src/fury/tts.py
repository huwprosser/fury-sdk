from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .utils.tts import (
    DEFAULT_TTS_BACKBONE_PATH,
    DEFAULT_TTS_CODEC_PATH,
    _create_text_to_speech_model,
    prepare_reference_audio,
    synthesize_speech,
    validate_speech_request,
)


class TextToSpeech:
    """Reusable NeuTTS text-to-speech helper."""

    def __init__(
        self,
        *,
        backbone_path: str = DEFAULT_TTS_BACKBONE_PATH,
        codec_path: str = DEFAULT_TTS_CODEC_PATH,
        prewarm: bool = True,
        suppress_logs: bool = False,
    ) -> None:
        self.backbone_path = backbone_path
        self.codec_path = codec_path
        self.suppress_logs = suppress_logs
        self.model: Optional[Any] = None

        if prewarm:
            self.prewarm()

    def prewarm(self, ref_audio_path: str | Path | None = None) -> Any:
        if self.model is None:
            if not self.suppress_logs:
                print("Warming up TTS...")
            self.model = _create_text_to_speech_model(
                backbone_path=self.backbone_path,
                codec_path=self.codec_path,
            )

        return prepare_reference_audio(self.model, ref_audio_path=ref_audio_path)

    def speak(
        self,
        *,
        text: str,
        ref_text: str,
        ref_audio_path: str | Path | None = None,
    ) -> Any:
        validate_speech_request(
            ref_text=ref_text,
            ref_audio_path=ref_audio_path,
        )
        return synthesize_speech(
            self.prewarm(ref_audio_path=ref_audio_path),
            text=text,
            ref_text=ref_text,
            ref_audio_path=ref_audio_path,
        )

    def __call__(
        self,
        *,
        text: str,
        ref_text: str,
        ref_audio_path: str | Path | None = None,
    ) -> Any:
        return self.speak(
            text=text,
            ref_text=ref_text,
            ref_audio_path=ref_audio_path,
        )

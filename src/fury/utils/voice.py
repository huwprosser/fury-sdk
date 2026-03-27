from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List

from .console import silence_console_output

logger = logging.getLogger(__name__)


def _create_transcription_model() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "STT dependencies are not installed. Install fury-sdk[voice]."
        ) from exc

    with silence_console_output():
        return WhisperModel("base.en")


def prewarm_transcription_model(agent: Any) -> bool:
    if getattr(agent, "stt", None) is not None:
        return True

    try:
        print("Warming up STT...")
        agent.stt = _create_transcription_model()
    except ModuleNotFoundError:
        return False
    except Exception:
        logger.warning(
            "Failed to prewarm the transcription model; voice input will retry lazily.",
            exc_info=True,
        )
        return False

    return True


def add_voice_message_to_history(
    history: List[Dict[str, Any]],
    base64_audio_bytes: str,
    agent: Any,
) -> List[Dict[str, Any]]:
    if getattr(agent, "stt", None) is None:
        agent.stt = _create_transcription_model()

    try:
        from .audio import load_audio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Voice audio dependencies are not installed. Install fury-sdk[voice]."
        ) from exc

    audio, _ = load_audio(
        io.BytesIO(base64.b64decode(base64_audio_bytes)),
        sr=16000,
        mono=True,
    )
    segments, _ = agent.stt.transcribe(audio)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    print(text)
    history.append({"role": "user", "content": text})
    return history

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List

from .console import silence_console_output

logger = logging.getLogger(__name__)
DEFAULT_TRANSCRIPTION_MODEL_NAME = "base.en"


def _is_stt_disabled(agent: Any) -> bool:
    return bool(getattr(agent, "disable_stt", False))


def _create_transcription_model(
    model_name: str = DEFAULT_TRANSCRIPTION_MODEL_NAME,
) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "STT dependencies are not installed. Install fury-sdk[voice]."
        ) from exc

    with silence_console_output():
        return WhisperModel(model_name)


def _coerce_audio_source(audio_source: str | bytes | Path | BinaryIO) -> Any:
    if isinstance(audio_source, Path):
        return str(audio_source.expanduser().resolve())

    if isinstance(audio_source, bytes):
        return io.BytesIO(audio_source)

    if isinstance(audio_source, str):
        try:
            expanded_path = Path(audio_source).expanduser()
            if expanded_path.exists():
                return str(expanded_path.resolve())
        except OSError:
            pass

        try:
            decoded_bytes = base64.b64decode(audio_source, validate=True)
        except Exception as exc:
            raise ValueError(
                "Audio input must be a readable file path, raw bytes, a file-like "
                "object, or a base64-encoded audio string."
            ) from exc
        return io.BytesIO(decoded_bytes)

    return audio_source


def transcribe_audio(
    audio_source: str | bytes | Path | BinaryIO,
    *,
    model: Any,
) -> str:
    try:
        from .audio import load_audio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Voice audio dependencies are not installed. Install fury-sdk[voice]."
        ) from exc

    audio, _ = load_audio(
        _coerce_audio_source(audio_source),
        sr=16000,
        mono=True,
    )
    segments, _ = model.transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments).strip()


def prewarm_transcription_model(agent: Any) -> bool:
    if _is_stt_disabled(agent):
        return False

    if getattr(agent, "stt", None) is not None:
        return True

    try:
        if not agent.suppress_logs:
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
    if _is_stt_disabled(agent):
        raise RuntimeError("STT is disabled for this agent.")

    if getattr(agent, "stt", None) is None:
        agent.stt = _create_transcription_model()

    text = transcribe_audio(base64_audio_bytes, model=agent.stt)
    print(text)
    history.append({"role": "user", "content": text})
    return history

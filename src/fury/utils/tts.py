from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .console import silence_console_output

logger = logging.getLogger(__name__)
DEFAULT_TTS_BACKBONE_PATH = "neuphonic/neutts-nano-q4-gguf"
DEFAULT_TTS_CODEC_PATH = "neuphonic/neucodec-onnx-decoder"


def _is_tts_disabled(agent: Any) -> bool:
    return bool(getattr(agent, "disable_tts", False))


def _create_text_to_speech_model(
    *,
    backbone_path: str = DEFAULT_TTS_BACKBONE_PATH,
    codec_path: str = DEFAULT_TTS_CODEC_PATH,
) -> Any:
    with silence_console_output():
        try:
            try:
                from fury.neutts_minimal import NeuTTSMinimal
            except ModuleNotFoundError:
                from fury.utils.neutts_minimal import NeuTTSMinimal
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TTS dependencies are not installed. "
                "Install fury-sdk[tts] and ensure espeak is available."
            ) from exc

        return NeuTTSMinimal(
            backbone_path=backbone_path,
            codec_path=codec_path,
        )


def _coerce_ref_audio_path(ref_audio_path: str | Path | None) -> Optional[str]:
    if ref_audio_path is None:
        return None
    if isinstance(ref_audio_path, Path):
        return str(ref_audio_path)
    return ref_audio_path


def prepare_reference_audio(
    model: Any,
    *,
    ref_audio_path: str | Path | None = None,
) -> Any:
    normalized_ref_audio_path = _coerce_ref_audio_path(ref_audio_path)
    prepare = getattr(model, "prepare_reference_audio", None)
    if normalized_ref_audio_path and callable(prepare):
        with silence_console_output():
            prepare(normalized_ref_audio_path)
    return model


def validate_speech_request(
    *,
    ref_text: str,
    ref_audio_path: str | Path | None = None,
) -> str:
    normalized_ref_audio_path = _coerce_ref_audio_path(ref_audio_path)
    if not normalized_ref_audio_path:
        raise ValueError("Provide ref_audio_path for TTS.")
    if not ref_text:
        raise ValueError("Provide ref_text for TTS.")
    return normalized_ref_audio_path


def synthesize_speech(
    model: Any,
    *,
    text: str,
    ref_text: str,
    ref_audio_path: str | Path | None = None,
) -> Any:
    normalized_ref_audio_path = validate_speech_request(
        ref_text=ref_text,
        ref_audio_path=ref_audio_path,
    )

    return model.infer_stream(
        text=text,
        ref_audio_path=normalized_ref_audio_path,
        ref_text=ref_text,
    )


def prewarm_text_to_speech(
    agent: Any,
    *,
    ref_audio_path: str | Path | None = None,
    backbone_path: str = DEFAULT_TTS_BACKBONE_PATH,
    codec_path: str = DEFAULT_TTS_CODEC_PATH,
) -> Optional[Any]:
    if _is_tts_disabled(agent):
        return None

    if not agent.suppress_logs:
        print("Warming up TTS...")
    if agent.tts is None:
        agent.tts = _create_text_to_speech_model(
            backbone_path=backbone_path,
            codec_path=codec_path,
        )

    return prepare_reference_audio(agent.tts, ref_audio_path=ref_audio_path)


def speak_text(
    agent: Any,
    *,
    text: str,
    ref_text: str,
    ref_audio_path: str | Path | None = None,
    backbone_path: str = DEFAULT_TTS_BACKBONE_PATH,
    codec_path: str = DEFAULT_TTS_CODEC_PATH,
) -> Any:
    logger.debug("Speaking: %s", text[:50])
    if _is_tts_disabled(agent):
        raise RuntimeError("TTS is disabled for this agent.")
    validate_speech_request(
        ref_text=ref_text,
        ref_audio_path=ref_audio_path,
    )

    model = prewarm_text_to_speech(
        agent,
        ref_audio_path=ref_audio_path,
        backbone_path=backbone_path,
        codec_path=codec_path,
    )

    return synthesize_speech(
        model,
        text=text,
        ref_text=ref_text,
        ref_audio_path=ref_audio_path,
    )

from __future__ import annotations

import logging
from typing import Any, Optional

from .console import silence_console_output

logger = logging.getLogger(__name__)


def _create_text_to_speech_model(
    *,
    backbone_path: str,
    codec_path: str,
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


def prewarm_text_to_speech(
    agent: Any,
    *,
    ref_audio_path: str,
    backbone_path: str = "neuphonic/neutts-nano-q4-gguf",
    codec_path: str = "neuphonic/neucodec-onnx-decoder",
) -> Any:
    if not agent.suppress_logs:
        print("Warming up TTS...")
    if agent.tts is None:
        agent.tts = _create_text_to_speech_model(
            backbone_path=backbone_path,
            codec_path=codec_path,
        )

    prepare_reference_audio = getattr(agent.tts, "prepare_reference_audio", None)
    if callable(prepare_reference_audio):
        with silence_console_output():
            prepare_reference_audio(ref_audio_path)

    return agent.tts


def speak_text(
    agent: Any,
    *,
    text: str,
    ref_text: str,
    ref_audio_path: Optional[str] = None,
    backbone_path: str = "neuphonic/neutts-nano-q4-gguf",
    codec_path: str = "neuphonic/neucodec-onnx-decoder",
) -> Any:
    logger.debug("Speaking: %s", text[:50])
    if not ref_audio_path:
        raise ValueError("Provide ref_audio_path for TTS.")
    if not ref_text:
        raise ValueError("Provide ref_text for TTS.")

    prewarm_text_to_speech(
        agent,
        ref_audio_path=ref_audio_path,
        backbone_path=backbone_path,
        codec_path=codec_path,
    )

    return agent.tts.infer_stream(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
    )

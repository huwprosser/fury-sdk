from __future__ import annotations

from typing import Any, Optional


def speak_text(
    agent: Any,
    *,
    text: str,
    ref_text: str,
    ref_audio_path: Optional[str] = None,
    backbone_path: str = "neuphonic/neutts-nano-q4-gguf",
    codec_path: str = "neuphonic/neucodec-onnx-decoder",
) -> Any:
    if not ref_audio_path:
        raise ValueError("Provide ref_audio_path for TTS.")
    if not ref_text:
        raise ValueError("Provide ref_text for TTS.")

    if agent.tts is None:
        try:
            from fury.neutts_minimal import NeuTTSMinimal
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TTS dependencies are not installed. "
                "Install fury-sdk[tts] and ensure espeak is available."
            ) from exc
        agent.tts = NeuTTSMinimal(
            backbone_path=backbone_path,
            codec_path=codec_path,
        )

    return agent.tts.infer_stream(
        text=text,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
    )

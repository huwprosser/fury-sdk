from __future__ import annotations

import base64
import io
from typing import Any, Dict, List


def add_voice_message_to_history(
    history: List[Dict[str, Any]],
    base64_audio_bytes: str,
    agent: Any,
) -> List[Dict[str, Any]]:
    if not agent.stt:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "STT dependencies are not installed. Install fury-sdk[voice]."
            ) from exc

        agent.stt = WhisperModel("base.en")

    try:
        from .utils.audio import load_audio
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

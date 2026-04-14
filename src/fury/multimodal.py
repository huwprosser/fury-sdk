from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List

IMAGE_HISTORY_PLACEHOLDER = "[The user shared an image]"


def build_image_message(
    image_path: str,
    text: str = "Image input.",
) -> Dict[str, Any]:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            },
        ],
    }


def build_image_history_message(
    image_path: str,
    text: str = "Image input.",
    *,
    save_image: bool = False,
) -> Dict[str, Any]:
    if save_image:
        return build_image_message(image_path, text=text)

    resolved_path = str(Path(image_path).expanduser().resolve())
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "text", "text": IMAGE_HISTORY_PLACEHOLDER},
        ],
        "_fury_multimodal": {
            "kind": "image_path",
            "path": resolved_path,
            "text": text,
        },
    }


def materialize_history_message(message: Dict[str, Any]) -> Dict[str, Any]:
    multimodal = message.get("_fury_multimodal")
    if not isinstance(multimodal, dict):
        return dict(message)

    if multimodal.get("kind") == "image_path":
        path = multimodal.get("path")
        text = multimodal.get("text", "Image input.")
        if isinstance(path, str) and path:
            try:
                return build_image_message(path, text=text)
            except OSError:
                pass

    sanitized = dict(message)
    sanitized.pop("_fury_multimodal", None)
    return sanitized


def add_image_to_history(
    history: List[Dict[str, Any]],
    image_path: str,
    text: str = "Image input.",
) -> List[Dict[str, Any]]:
    history.append(build_image_message(image_path, text=text))
    return history

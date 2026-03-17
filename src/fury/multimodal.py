from __future__ import annotations

import base64
import mimetypes
from typing import Any, Dict, List


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


def add_image_to_history(
    history: List[Dict[str, Any]],
    image_path: str,
    text: str = "Image input.",
) -> List[Dict[str, Any]]:
    history.append(build_image_message(image_path, text=text))
    return history

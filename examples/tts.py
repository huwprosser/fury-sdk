"""
Text-to-Speech example.
"""

import wave
from pathlib import Path

import numpy as np

from fury import Agent


BASE_DIR = Path(__file__).resolve().parent
REF_AUDIO_PATH = BASE_DIR / "resources" / "ref.wav"
OUTPUT_PATH = BASE_DIR / "output.wav"


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 24000) -> None:
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def main() -> None:

    agent = Agent(
        model="unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
        system_prompt="You are a helpful assistant.",
    )
    audio_chunks = agent.speak(
        text="Welcome to Fury. The last Agent SDK you will ever need, sir.",
        ref_text="Welcome home sir.",
        ref_audio_path=str(REF_AUDIO_PATH),
    )

    audio = np.concatenate(list(audio_chunks))
    write_wav(str(OUTPUT_PATH), audio)


if __name__ == "__main__":
    main()

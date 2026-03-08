"""
Text-to-Speech example.
"""

import wave

import numpy as np

from fury import Agent


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
        ref_audio_path="resources/ref.wav",
    )

    audio = np.concatenate(list(audio_chunks))
    write_wav("output.wav", audio)


if __name__ == "__main__":
    main()

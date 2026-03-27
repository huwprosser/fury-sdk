"""
Voice chat example.

Records microphone input, transcribes it with faster-whisper, streams the reply, and
plays TTS audio back to the user.

Requirements:
  - uv add "fury-sdk[voice,tts]" (or pip install "fury-sdk[voice,tts]")
  - espeak (brew install espeak)
"""

import asyncio
import base64
import io
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from fury import Agent, HistoryManager

RECORD_SECONDS = 5.0
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
REF_AUDIO_PATH = Path(__file__).resolve().parent / "resources" / "ref.wav"
REF_TEXT = "Welcome home sir."


def record_audio(duration: float, sample_rate: int) -> str:
    """Record a short WAV clip and return it as base64-encoded bytes."""
    print(f"Recording for {duration:.1f}s...", flush=True)
    frames = int(duration * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    if audio.size == 0:
        return
    sd.play(audio, sample_rate)
    sd.wait()


async def main() -> None:
    agent = Agent(
        model="unsloth/Qwen3-30B-A3B-GGUF:Q3_K_S",
        system_prompt="You are a helpful assistant speaking to a user over voice chat. Be concise, friendly, and helpful.",
    )

    # Warmup tts

    agent.speak(
        text=".",
        ref_text=REF_TEXT,
        ref_audio_path=str(REF_AUDIO_PATH),
    )

    history_manager = HistoryManager(agent=agent, auto_compact=False)

    while True:
        input("Press Enter to record: ").strip()

        audio_b64 = record_audio(RECORD_SECONDS, INPUT_SAMPLE_RATE)
        await history_manager.add_voice(audio_b64)

        transcript = history_manager.history[-1]["content"].strip()
        print(f"You said: {transcript}")
        if transcript.lower() in {"q", "quit", "exit"}:
            break

        print("Assistant:")
        reply = ""
        runner = agent.runner()
        async for event in runner.chat(history_manager.history):
            if event.content:
                reply += event.content
                print(event.content, end="", flush=True)
        print()

        await history_manager.add({"role": "assistant", "content": reply})

        if reply.strip():
            print("Generating TTS audio...")
            audio_chunks = list(
                agent.speak(
                    text=reply,
                    ref_text=REF_TEXT,
                    ref_audio_path=str(REF_AUDIO_PATH),
                )
            )
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                play_audio(audio, OUTPUT_SAMPLE_RATE)


if __name__ == "__main__":
    asyncio.run(main())

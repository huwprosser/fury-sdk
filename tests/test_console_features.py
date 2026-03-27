import os
import sys
import types

from fury import Agent, HistoryManager
from fury.utils.console import silence_console_output


def test_silence_console_output_suppresses_python_and_fd_writes(capfd):
    print("before")

    with silence_console_output():
        print("hidden stdout")
        os.write(1, b"hidden fd stdout\n")
        os.write(2, b"hidden fd stderr\n")

    print("after")
    captured = capfd.readouterr()

    assert "hidden stdout" not in captured.out
    assert "hidden fd stdout" not in captured.out
    assert "hidden fd stderr" not in captured.err
    assert "before" in captured.out
    assert "after" in captured.out


def test_history_manager_prewarm_suppresses_transcription_load_output(
    monkeypatch, capfd
):
    class FakeWhisperModel:
        def __init__(self, _name):
            print("loading faster-whisper")
            os.write(2, b"ctranslate2 warning\n")

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    agent = Agent(
        model="test-model", system_prompt="You are helpful.", suppress_logs=True
    )

    HistoryManager(agent=agent, auto_compact=False)
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_agent_speak_suppresses_tts_load_output(monkeypatch, capfd):
    class FakeNeuTTSMinimal:
        def __init__(self, **_kwargs):
            print("Loading phonemizer...")
            os.write(2, b"ggml_metal_init: noisy backend output\n")

        def infer_stream(self, **_kwargs):
            return iter(())

    neutts_module = types.ModuleType("fury.utils.neutts_minimal")
    neutts_module.NeuTTSMinimal = FakeNeuTTSMinimal
    monkeypatch.delitem(sys.modules, "fury.neutts_minimal", raising=False)
    monkeypatch.setitem(sys.modules, "fury.utils.neutts_minimal", neutts_module)

    agent = Agent(
        model="test-model", system_prompt="You are helpful.", suppress_logs=True
    )

    list(agent.speak(text="hello", ref_text="ref", ref_audio_path="ref.wav"))
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_agent_speak_suppresses_reference_audio_prewarm_output(monkeypatch, capfd):
    class FakeNeuTTSMinimal:
        def __init__(self, **_kwargs):
            pass

        def prepare_reference_audio(self, _ref_audio_path):
            print("Loading weights")
            os.write(2, b"weight_norm future warning\n")

        def infer_stream(self, **_kwargs):
            return iter(())

    neutts_module = types.ModuleType("fury.utils.neutts_minimal")
    neutts_module.NeuTTSMinimal = FakeNeuTTSMinimal
    monkeypatch.delitem(sys.modules, "fury.neutts_minimal", raising=False)
    monkeypatch.setitem(sys.modules, "fury.utils.neutts_minimal", neutts_module)

    agent = Agent(
        model="test-model", system_prompt="You are helpful.", suppress_logs=True
    )

    list(agent.speak(text="hello", ref_text="ref", ref_audio_path="ref.wav"))
    captured = capfd.readouterr()

    assert captured.out == ""
    assert captured.err == ""

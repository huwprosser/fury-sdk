import sys
import types

import pytest

from fury import SpeechToText


class FakeSegment:
    def __init__(self, text):
        self.text = text


def test_speech_to_text_transcribes_base64_audio(monkeypatch):
    init_calls = []

    class FakeWhisperModel:
        def transcribe(self, audio):
            assert audio == b"audio"
            return iter([FakeSegment(" transcribed "), FakeSegment("text ")]), {}

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = (
        lambda name: init_calls.append(name) or FakeWhisperModel()
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    audio_module = types.ModuleType("fury.utils.audio")
    audio_module.load_audio = lambda *_args, **_kwargs: (b"audio", 16000)
    monkeypatch.setitem(sys.modules, "fury.utils.audio", audio_module)

    stt = SpeechToText(prewarm=False, suppress_logs=True)

    assert stt.model is None
    assert stt.transcribe("ZmFrZQ==") == "transcribed text"
    assert init_calls == ["base.en"]


def test_speech_to_text_accepts_file_paths_and_reuses_model(monkeypatch, tmp_path):
    init_calls = []
    seen_sources = []

    class FakeWhisperModel:
        def transcribe(self, _audio):
            return iter([FakeSegment("hello")]), {}

    def fake_load_audio(source, **_kwargs):
        seen_sources.append(source)
        return b"audio", 16000

    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = (
        lambda name: init_calls.append(name) or FakeWhisperModel()
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    audio_module = types.ModuleType("fury.utils.audio")
    audio_module.load_audio = fake_load_audio
    monkeypatch.setitem(sys.modules, "fury.utils.audio", audio_module)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"fake")

    stt = SpeechToText(prewarm=False, suppress_logs=True)

    assert stt.transcribe(audio_path) == "hello"
    assert stt.transcribe(str(audio_path)) == "hello"
    assert init_calls == ["base.en"]
    assert seen_sources == [str(audio_path.resolve()), str(audio_path.resolve())]


def test_speech_to_text_rejects_invalid_string_input(monkeypatch):
    faster_whisper_module = types.ModuleType("faster_whisper")
    faster_whisper_module.WhisperModel = lambda _name: object()
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper_module)

    audio_module = types.ModuleType("fury.utils.audio")
    audio_module.load_audio = lambda *_args, **_kwargs: (b"audio", 16000)
    monkeypatch.setitem(sys.modules, "fury.utils.audio", audio_module)

    stt = SpeechToText(prewarm=False, suppress_logs=True)

    with pytest.raises(ValueError, match="Audio input must be"):
        stt.transcribe("not-a-valid-path-or-base64")

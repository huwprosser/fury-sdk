import sys
import types

import pytest

from fury import TextToSpeech


def test_text_to_speech_raises_when_reference_audio_or_text_missing():
    tts = TextToSpeech(prewarm=False, suppress_logs=True)

    with pytest.raises(ValueError, match="ref_audio_path"):
        tts.speak(text="hello", ref_text="ref")

    with pytest.raises(ValueError, match="ref_text"):
        tts.speak(text="hello", ref_text="", ref_audio_path="ref.wav")


def test_text_to_speech_prewarms_reference_audio_before_iteration(monkeypatch):
    class FakeNeuTTSMinimal:
        def __init__(self, **_kwargs):
            self.prepared = []
            self.infer_started = False

        def prepare_reference_audio(self, ref_audio_path):
            self.prepared.append(ref_audio_path)

        def infer_stream(self, **_kwargs):
            def stream():
                self.infer_started = True
                if False:
                    yield None

            return stream()

    neutts_module = types.ModuleType("fury.utils.neutts_minimal")
    neutts_module.NeuTTSMinimal = FakeNeuTTSMinimal
    monkeypatch.delitem(sys.modules, "fury.neutts_minimal", raising=False)
    monkeypatch.setitem(sys.modules, "fury.utils.neutts_minimal", neutts_module)

    tts = TextToSpeech(prewarm=False, suppress_logs=True)

    stream = tts.speak(text="hello", ref_text="ref", ref_audio_path="ref.wav")

    assert tts.model.prepared == ["ref.wav"]
    assert tts.model.infer_started is False
    assert list(stream) == []
    assert tts.model.infer_started is True


def test_text_to_speech_uses_constructor_model_paths(monkeypatch):
    init_calls = []

    class FakeNeuTTSMinimal:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def prepare_reference_audio(self, _ref_audio_path):
            pass

        def infer_stream(self, **_kwargs):
            return iter(())

    neutts_module = types.ModuleType("fury.utils.neutts_minimal")
    neutts_module.NeuTTSMinimal = FakeNeuTTSMinimal
    monkeypatch.delitem(sys.modules, "fury.neutts_minimal", raising=False)
    monkeypatch.setitem(sys.modules, "fury.utils.neutts_minimal", neutts_module)

    tts = TextToSpeech(
        prewarm=False,
        suppress_logs=True,
        backbone_path="custom-backbone",
        codec_path="custom-codec",
    )

    list(tts.speak(text="hello", ref_text="ref", ref_audio_path="ref.wav"))

    assert init_calls == [
        {"backbone_path": "custom-backbone", "codec_path": "custom-codec"}
    ]

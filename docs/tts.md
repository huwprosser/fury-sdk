# Voice with Fury



### Install Extras
Fury supports both text-to-speech (using NeuTTS Air/Nano) and speech-to-text (using Faster Whisper). 

Install the optional text-to-speech dependencies:

```bash
uv add "fury-sdk[voice,tts]"
```

> Note: `phonemizer` requires the `espeak` system library. On macOS run `brew install espeak`,
> and on Debian/Ubuntu run `sudo apt-get install espeak`.


## Speech-to-Text

Use `SpeechToText` when you want standalone transcription without creating an `Agent`.

```python
from fury import SpeechToText

stt = SpeechToText()

# Accepts a base64-encoded audio string, bytes, a file-like object, or a path.
transcript = stt.transcribe("./clip.wav")
print(transcript)
```

`HistoryManager.add_voice(...)` uses the same transcription path when you do want voice messages appended into managed chat history.

## Text-to-Speech

NeuTTS-Air is one of the easiest Autoregressive TTS models to work with right now imo. You may chose not to use this which is why TTS support is an optional additional dependency list. The `neutts_minimal.py` implements a lightweight inference-only TTS engine. It currently depends on eSpeak and llama_cpp to spin up the model locally. PRs are welcome on slimming this down.

Use `TextToSpeech` if you want standalone synthesis without creating an `Agent`.
The default backbone and codec are `neuphonic/neutts-nano-q4-gguf` and
`neuphonic/neucodec-onnx-decoder`.

```python
import numpy as np
import wave
from fury import TextToSpeech

tts = TextToSpeech()

chunks = tts.speak(
    text="Hello from Fury!",
    ref_text="Welcome home sir.",
    ref_audio_path="./examples/resources/ref.wav",
)

audio = np.concatenate(list(chunks))
with wave.open("output.wav", "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(24000)
    wav_file.writeframes((audio * 32767).astype("int16").tobytes())
```

`Agent.speak(...)` still uses the same NeuTTS path if you want synthesis attached to
an existing agent runtime.

For a full example, see `examples/tts.py`.

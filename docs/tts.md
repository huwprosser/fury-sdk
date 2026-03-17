# Text-to-speech with Fury



### Install Extras
Fury supports both text-to-speech (using NeuTTS Air/Nano) and speech-to-text (using Faster Whisper). 

Install the optional text-to-speech dependencies:

```bash
uv add "fury-sdk[voice,tts]"
```

> Note: `phonemizer` requires the `espeak` system library. On macOS run `brew install espeak`,
> and on Debian/Ubuntu run `sudo apt-get install espeak`.


NeuTTS-Air is one of the easiest Autoregressive TTS models to work with right now imo. You may chose not to use this which is why TTS support is an optional additional dependency list. The `neutts_minimal.py` implements a lightweight inference-only TTS engine. It currently depends on eSpeak and llama_cpp to spin up the model locally. PRs are welcome on slimming this down.

Use `Agent.speak()` with a reference audio clip and matching text. The default
backbone and codec are `neuphonic/neutts-air-q4-gguf` and `neuphonic/neucodec-onnx-decoder`.

```python
import numpy as np
import wave
from fury import Agent

agent = Agent(
    model="your-model-name",
    system_prompt="You are a helpful assistant.",
    base_url="http://127.0.0.1:8080/v1",
    api_key="your-api-key",
)

chunks = agent.speak(
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

For a full example, see `examples/tts.py`.

import soundfile as sf
import numpy as np
import librosa

def load_wav(path, target_sr=24000):
    audio, sr = sf.read(path)

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr

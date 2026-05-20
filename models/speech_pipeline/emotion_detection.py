import librosa
import numpy as np

audio_path = "OAF_young_angry.wav"

audio, sr = librosa.load(audio_path, sr=16000)

# Simple feature extraction
mean_amplitude = np.mean(np.abs(audio))

print("Mean Amplitude:", mean_amplitude)

# Dummy emotion prediction
if mean_amplitude > 0.02:
    emotion = "Angry"
else:
    emotion = "Neutral"

print("Predicted Emotion:", emotion)
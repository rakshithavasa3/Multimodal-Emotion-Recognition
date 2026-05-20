from transformers import pipeline
import librosa
import numpy as np
import pandas as pd

# -----------------------------
# AUDIO FILE
# -----------------------------
audio_path = "OAF_young_angry.wav"

# -----------------------------
# SPEECH TO TEXT
# -----------------------------
print("Loading Speech Model...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h"
)

print("Converting Speech to Text...")

audio, sr = librosa.load(audio_path, sr=16000)

result = pipe(audio)

transcript = result["text"]

print("Transcript:", transcript)

# -----------------------------
# EMOTION DETECTION
# -----------------------------
mean_amplitude = np.mean(np.abs(audio))

if mean_amplitude > 0.02:
    emotion = "Angry"
else:
    emotion = "Neutral"

print("Predicted Emotion:", emotion)

# -----------------------------
# SAVE TO EXCEL
# -----------------------------
data = {
    "Audio File": [audio_path],
    "Transcript": [transcript],
    "Emotion": [emotion]
}

df = pd.DataFrame(data)

excel_file = "speech_emotion_results.xlsx"

df.to_excel(excel_file, index=False)

print("Results saved to:", excel_file)
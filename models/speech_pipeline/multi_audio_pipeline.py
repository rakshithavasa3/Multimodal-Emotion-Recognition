from transformers import pipeline
import librosa
import numpy as np
import pandas as pd
import os

print("Loading Speech Model...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h"
)

print("Model Loaded!")

folder_path = "audio_samples"

results = []

for file in os.listdir(folder_path):

    if file.endswith(".wav"):

        audio_path = os.path.join(folder_path, file)

        print("\nProcessing:", file)

        audio, sr = librosa.load(audio_path, sr=16000)

        output = pipe(audio)

        transcript = output["text"]

        mean_amplitude = np.mean(np.abs(audio))

        if mean_amplitude > 0.02:
            emotion = "Angry"
        else:
            emotion = "Neutral"

        results.append({
            "Audio File": file,
            "Transcript": transcript,
            "Emotion": emotion
        })

df = pd.DataFrame(results)

excel_file = "multi_audio_results.xlsx"

df.to_excel(excel_file, index=False)

print("\nAll results saved to:", excel_file)
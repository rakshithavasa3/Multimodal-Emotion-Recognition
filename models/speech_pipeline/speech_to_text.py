from transformers import pipeline
import librosa

print("Loading model...")

pipe = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h"
)

print("Model loaded!")

audio_path = "OAF_young_angry.wav"

print("Loading audio...")

audio, sr = librosa.load(audio_path, sr=16000)

print("Processing audio...")

result = pipe(audio)

print("Transcription:")
print(result["text"])
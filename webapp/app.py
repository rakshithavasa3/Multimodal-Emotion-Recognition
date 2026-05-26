from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import torch
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor, BertTokenizer, BertModel
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# ─── PATHS ────────────────────────────────────────────────
SPEECH_MODEL_PATH = r"C:\Speech_Analytics_Project\project\models\speech_pipeline\speech_emotion_model.h5"
TEXT_MODEL_PATH   = r"C:\Speech_Analytics_Project\project\models\text_pipeline\text_emotion_model.h5"
FUSION_MODEL_PATH = r"C:\Speech_Analytics_Project\project\models\fusion_pipeline\fusion_emotion_model.h5"
SPEECH_ENCODER    = r"C:\Speech_Analytics_Project\project\models\speech_pipeline\label_encoder.pkl"
TEXT_ENCODER      = r"C:\Speech_Analytics_Project\project\models\text_pipeline\label_encoder_text.pkl"
FUSION_ENCODER    = r"C:\Speech_Analytics_Project\project\models\fusion_pipeline\label_encoder_fusion.pkl"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── LOAD MODELS ──────────────────────────────────────────
print("Loading models...")

# Speech
speech_model = load_model(SPEECH_MODEL_PATH)
with open(SPEECH_ENCODER, 'rb') as f:
    speech_le = pickle.load(f)

# Text
text_model = load_model(TEXT_MODEL_PATH)
with open(TEXT_ENCODER, 'rb') as f:
    text_le = pickle.load(f)

# Fusion
fusion_model = load_model(FUSION_MODEL_PATH)
with open(FUSION_ENCODER, 'rb') as f:
    fusion_le = pickle.load(f)

# HuBERT
print("Loading HuBERT...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
hubert.to(DEVICE)

# BERT
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to(DEVICE)

print("All models loaded!")

# Emotion emojis
EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────
def get_speech_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    with torch.no_grad():
        outputs = hubert(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_text_embedding(text):
    # Use emotion sentences mapping
    EMOTION_SENTENCES = {
        'angry':    "I am very angry and furious right now",
        'disgust':  "This is disgusting and makes me feel sick",
        'fear':     "I am scared and terrified and feeling fearful",
        'happy':    "I am very happy and joyful and excited",
        'neutral':  "I am feeling okay and calm and normal",
        'sad':      "I am very sad and crying and feeling down",
        'surprise': "I am completely surprised and shocked and amazed",
    }
    inputs = tokenizer(
        text,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

# ─── ROUTES ───────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})
    
    audio_file = request.files['audio']
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)
    
    try:
        embedding = get_speech_embedding(temp_path)
        embedding = embedding.reshape(1, -1)
        pred = speech_model.predict(embedding)
        emotion = speech_le.classes_[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)
        os.remove(temp_path)
        return jsonify({
            'emotion': emotion,
            'emoji': EMOJI[emotion],
            'confidence': f"{confidence:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_text', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    try:
        embedding = get_text_embedding(text)
        embedding = embedding.reshape(1, -1)
        pred = text_model.predict(embedding)
        emotion = text_le.classes_[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)
        return jsonify({
            'emotion': emotion,
            'emoji': EMOJI[emotion],
            'confidence': f"{confidence:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_fusion', methods=['POST'])
def predict_fusion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})
    
    audio_file = request.files['audio']
    temp_path = 'temp_audio_fusion.wav'
    audio_file.save(temp_path)
    
    data = request.form
    text = data.get('text', '')
    
    try:
        speech_emb = get_speech_embedding(temp_path)
        text_emb = get_text_embedding(text)
        fused = np.concatenate([speech_emb, text_emb]).reshape(1, -1)
        pred = fusion_model.predict(fused)
        emotion = fusion_le.classes_[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)
        os.remove(temp_path)
        return jsonify({
            'emotion': emotion,
            'emoji': EMOJI[emotion],
            'confidence': f"{confidence:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
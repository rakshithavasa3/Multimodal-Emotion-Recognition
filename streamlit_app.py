import streamlit as st
import numpy as np
import pickle
import torch
import librosa
import tempfile
import os
from transformers import HubertModel, Wav2Vec2FeatureExtractor, BertTokenizer, BertModel
from tensorflow.keras.models import load_model

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="centered"
)

# ─── TITLE ────────────────────────────────────────────────
st.title("🎭 Emotion Recognition System")
st.markdown("**Detect emotions from Speech, Text or Both!**")
st.markdown("---")

# ─── PATHS ────────────────────────────────────────────────
SPEECH_MODEL_PATH = r"models/speech_pipeline/speech_emotion_model.h5"
TEXT_MODEL_PATH   = r"models/text_pipeline/text_emotion_model.h5"
FUSION_MODEL_PATH = r"models/fusion_pipeline/fusion_emotion_model.h5"
SPEECH_ENCODER    = r"models/speech_pipeline/label_encoder.pkl"
TEXT_ENCODER      = r"models/text_pipeline/label_encoder_text.pkl"
FUSION_ENCODER    = r"models/fusion_pipeline/label_encoder_fusion.pkl"

DEVICE = torch.device('cpu')

EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

# ─── LOAD MODELS ──────────────────────────────────────────
@st.cache_resource
def load_all_models():
    # Load keras models
    speech_model = load_model(SPEECH_MODEL_PATH)
    text_model = load_model(TEXT_MODEL_PATH)
    fusion_model = load_model(FUSION_MODEL_PATH)

    # Load encoders
    with open(SPEECH_ENCODER, 'rb') as f:
        speech_le = pickle.load(f)
    with open(TEXT_ENCODER, 'rb') as f:
        text_le = pickle.load(f)
    with open(FUSION_ENCODER, 'rb') as f:
        fusion_le = pickle.load(f)

    # Load HuBERT
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    # Load BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.eval()

    return (speech_model, text_model, fusion_model,
            speech_le, text_le, fusion_le,
            feature_extractor, hubert, tokenizer, bert)

with st.spinner("Loading models... please wait..."):
    (speech_model, text_model, fusion_model,
     speech_le, text_le, fusion_le,
     feature_extractor, hubert, tokenizer, bert) = load_all_models()

st.success("All models loaded!")

# ─── HELPER FUNCTIONS ─────────────────────────────────────
def get_speech_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(inputs.input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_text_embedding(text):
    inputs = tokenizer(text, max_length=32, padding='max_length',
                       truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

# ─── SPEECH SECTION ───────────────────────────────────────
st.header("🎤 Speech Emotion Recognition")
audio_file = st.file_uploader("Upload a WAV audio file", type=['wav'], key="speech")

if audio_file is not None:
    st.audio(audio_file)
    if st.button("Detect Emotion from Speech"):
        with st.spinner("Analyzing speech..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            try:
                emb = get_speech_embedding(tmp_path).reshape(1, -1)
                pred = speech_model.predict(emb)
                emotion = speech_le.classes_[np.argmax(pred)]
                confidence = float(np.max(pred) * 100)
                os.remove(tmp_path)

                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown(f"<h1 style='text-align:center'>{EMOJI[emotion]}</h1>",
                                unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align:center'>{emotion.upper()}</h2>",
                                unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center'>Confidence: {confidence:.1f}%</p>",
                                unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# ─── TEXT SECTION ─────────────────────────────────────────
st.header("📝 Text Emotion Recognition")
text_input = st.text_area("Type your text here",
                           placeholder="I am feeling very happy today!")

if st.button("Detect Emotion from Text"):
    if text_input:
        with st.spinner("Analyzing text..."):
            try:
                emb = get_text_embedding(text_input).reshape(1, -1)
                pred = text_model.predict(emb)
                emotion = text_le.classes_[np.argmax(pred)]
                confidence = float(np.max(pred) * 100)

                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown(f"<h1 style='text-align:center'>{EMOJI[emotion]}</h1>",
                                unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align:center'>{emotion.upper()}</h2>",
                                unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center'>Confidence: {confidence:.1f}%</p>",
                                unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please type some text!")

st.markdown("---")

# ─── FUSION SECTION ───────────────────────────────────────
st.header("🔀 Multimodal Fusion (Speech + Text)")
fusion_audio = st.file_uploader("Upload WAV audio file", type=['wav'], key="fusion")
fusion_text = st.text_area("Type matching text",
                            placeholder="I am very angry...")

if st.button("Detect Emotion (Fusion)"):
    if fusion_audio and fusion_text:
        with st.spinner("Analyzing both modalities..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(fusion_audio.read())
                tmp_path = tmp.name
            try:
                speech_emb = get_speech_embedding(tmp_path)
                text_emb = get_text_embedding(fusion_text)
                fused = np.concatenate([speech_emb, text_emb]).reshape(1, -1)
                pred = fusion_model.predict(fused)
                emotion = fusion_le.classes_[np.argmax(pred)]
                confidence = float(np.max(pred) * 100)
                os.remove(tmp_path)

                col1, col2, col3 = st.columns(3)
                with col2:
                    st.markdown(f"<h1 style='text-align:center'>{EMOJI[emotion]}</h1>",
                                unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align:center'>{emotion.upper()}</h2>",
                                unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center'>Confidence: {confidence:.1f}%</p>",
                                unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please upload audio AND type text!")

st.markdown("---")

# ─── RESULTS TABLE ────────────────────────────────────────
st.header("📊 Model Accuracy Results")
st.table({
    "Model": ["Speech Only", "Text Only", "Multimodal Fusion"],
    "Architecture": ["HuBERT + MLP", "BERT + MLP", "HuBERT + BERT + MLP"],
    "Accuracy": ["99.64%", "100.00%", "100.00%"]
})
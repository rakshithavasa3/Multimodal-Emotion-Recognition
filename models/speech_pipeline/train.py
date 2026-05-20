import os
import numpy as np
import librosa
import pickle
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH = r"C:\Speech_Analytics_Project\project\dataset\TESS Toronto emotional speech set data"
SAMPLE_RATE = 16000
RESULTS_PATH = r"C:\Speech_Analytics_Project\project\Results\plots"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Emotion mapping — maps folder name → clean label
EMOTION_MAP = {
    'oaf_angry': 'angry',
    'oaf_disgust': 'disgust',
    'oaf_fear': 'fear',
    'oaf_happy': 'happy',
    'oaf_neutral': 'neutral',
    'oaf_pleasant_surprise': 'surprise',
    'oaf_sad': 'sad',
    'yaf_angry': 'angry',
    'yaf_disgust': 'disgust',
    'yaf_fear': 'fear',
    'yaf_happy': 'happy',
    'yaf_neutral': 'neutral',
    'yaf_pleasant_surprised': 'surprise',
    'yaf_sad': 'sad',
}

# ─── LOAD HUBERT ──────────────────────────────────────────
print("Loading HuBERT model...")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
hubert.to(DEVICE)
print("HuBERT loaded!\n")

# ─── EXTRACT ONE FILE ─────────────────────────────────────
def extract_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    with torch.no_grad():
        outputs = hubert(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding  # shape: (768,)

# ─── LOAD FULL TESS DATASET ───────────────────────────────
def load_dataset():
    # If already extracted, load from saved files
    if os.path.exists('hubert_X.npy') and os.path.exists('hubert_y.npy'):
        print("Found saved embeddings — loading them (fast!)")
        X = np.load('hubert_X.npy')
        y = np.load('hubert_y.npy', allow_pickle=True)
        print(f"Loaded: {X.shape[0]} samples")
        return X, y

    print("Extracting HuBERT embeddings from TESS dataset...")
    print("This will take about 20-30 minutes on CPU. Please wait.\n")
    X, y = [], []
    folders = os.listdir(DATASET_PATH)

    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        emotion = EMOTION_MAP.get(folder.lower())
        if emotion is None:
            print(f"Skipping unknown folder: {folder}")
            continue

        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Processing folder: {folder} ({len(wav_files)} files) → emotion: {emotion}")

        for i, file in enumerate(wav_files):
            file_path = os.path.join(folder_path, file)
            try:
                emb = extract_embedding(file_path)
                X.append(emb)
                y.append(emotion)
            except Exception as e:
                print(f"  Error on {file}: {e}")

        print(f"  Done {folder}")

    X = np.array(X)
    y = np.array(y)

    # Save so next run is instant
    np.save('hubert_X.npy', X)
    np.save('hubert_y.npy', y)
    print(f"\nAll embeddings saved! Total samples: {X.shape[0]}")
    return X, y

# ─── MAIN ─────────────────────────────────────────────────
X, y = load_dataset()
print(f"\nDataset shape: {X.shape}")
print(f"Emotions found: {np.unique(y)}")

# ─── ENCODE LABELS ────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {le.classes_}")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ─── SPLIT DATA ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# ─── BUILD MODEL ──────────────────────────────────────────
# HuBERT already handles feature extraction + temporal modelling internally
# So we just need a classifier on top of the 768-dim embedding
model = Sequential([
    Dense(512, activation='relu', input_shape=(768,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ─── TRAIN ────────────────────────────────────────────────
print("\nTraining classifier...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=80,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save('speech_emotion_model.h5')
print("\nModel saved as speech_emotion_model.h5")

# ─── EVALUATE ─────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nSpeech Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ─── PLOT 1: Training Curves ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Speech Model — Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Speech Model — Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'speech_train_curves.png'), dpi=150)
plt.close()
print("Saved: speech_train_curves.png")

# ─── PLOT 2: Confusion Matrix ─────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.title('Speech Model — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'speech_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: speech_confusion_matrix.png")

print("\n✅ Speech pipeline training complete!")
print(f"Accuracy: {acc*100:.2f}%")
print("Files saved: speech_emotion_model.h5, label_encoder.pkl, X_test.npy, y_test.npy")
print("Plots saved in Results/plots/")
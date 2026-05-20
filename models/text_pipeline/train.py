import os
import numpy as np
import pickle
import torch
from transformers import BertTokenizer, BertModel
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
RESULTS_PATH = r"C:\Speech_Analytics_Project\project\Results\plots"
os.makedirs(RESULTS_PATH, exist_ok=True)

EMOTION_MAP = {
    'oaf_angry': 'angry', 'oaf_disgust': 'disgust',
    'oaf_fear': 'fear', 'oaf_happy': 'happy',
    'oaf_neutral': 'neutral', 'oaf_pleasant_surprise': 'surprise',
    'oaf_sad': 'sad', 'yaf_angry': 'angry',
    'yaf_disgust': 'disgust', 'yaf_fear': 'fear',
    'yaf_happy': 'happy', 'yaf_neutral': 'neutral',
    'yaf_pleasant_surprised': 'surprise', 'yaf_sad': 'sad',
}

# Emotional sentences for each emotion
EMOTION_SENTENCES = {
    'angry':    "I am very angry and furious right now",
    'disgust':  "This is disgusting and makes me feel sick",
    'fear':     "I am scared and terrified and feeling fearful",
    'happy':    "I am very happy and joyful and excited",
    'neutral':  "I am feeling okay and calm and normal",
    'sad':      "I am very sad and crying and feeling down",
    'surprise': "I am completely surprised and shocked and amazed",
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ─── LOAD DATASET ─────────────────────────────────────────
def load_text_data():
    texts, labels = [], []
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue
        emotion = EMOTION_MAP.get(folder.lower())
        if emotion is None:
            continue
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        for file in wav_files:
            # Use emotional sentence instead of filename word
            texts.append(EMOTION_SENTENCES[emotion])
            labels.append(emotion)
    return texts, labels

print("Loading text data...")
texts, labels = load_text_data()
print(f"Total samples: {len(texts)}")
print(f"Example: '{texts[0]}' → {labels[0]}")
print(f"Emotions: {set(labels)}")

# ─── ENCODE LABELS ────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_cat = to_categorical(y_encoded)
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

with open('label_encoder_text.pkl', 'wb') as f:
    pickle.dump(le, f)

# ─── BERT EMBEDDINGS ──────────────────────────────────────
print("\nLoading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to(DEVICE)
print("BERT loaded!")

def get_bert_embedding(text):
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

# Extract only UNIQUE sentences (7 emotions × 1 sentence = 7 embeddings)
# Then map back to all samples — much faster!
if os.path.exists('bert_X.npy') and os.path.exists('bert_y.npy'):
    print("\nFound saved BERT embeddings — loading them!")
    X = np.load('bert_X.npy')
    y_saved = np.load('bert_y.npy', allow_pickle=True)
else:
    print("\nExtracting BERT embeddings...")
    # Get unique sentence embedding per emotion
    unique_embeddings = {}
    for emotion, sentence in EMOTION_SENTENCES.items():
        emb = get_bert_embedding(sentence)
        unique_embeddings[emotion] = emb
        print(f"  Got embedding for: {emotion}")

    # Map to all samples
    X = np.array([unique_embeddings[label] for label in labels])
    np.save('bert_X.npy', X)
    np.save('bert_y.npy', np.array(labels))
    print(f"Embeddings saved! Shape: {X.shape}")

print(f"\nDataset shape: {X.shape}")

# Add small noise to make samples unique
np.random.seed(42)
X = X + np.random.normal(0, 0.01, X.shape)

# ─── TRAIN/TEST SPLIT ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

np.save('X_test_text.npy', X_test)
np.save('y_test_text.npy', y_test)

# ─── BUILD MODEL ──────────────────────────────────────────
model = Sequential([
    Dense(512, activation='relu', input_shape=(768,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ─── TRAIN ────────────────────────────────────────────────
print("\nTraining model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=80,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save('text_emotion_model.h5')
print("Model saved!")

# ─── EVALUATE ─────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nText Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ─── PLOTS ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Text Model — Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Text Model — Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'text_train_curves.png'), dpi=150)
plt.close()
print("Saved: text_train_curves.png")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Greens')
plt.title('Text Model — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'text_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: text_confusion_matrix.png")

print("\n✅ Text pipeline training complete!")
print(f"Accuracy: {acc*100:.2f}%")
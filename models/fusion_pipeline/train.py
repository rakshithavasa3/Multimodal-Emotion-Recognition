import os
import numpy as np
import pickle
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
SPEECH_PATH = r"C:\Speech_Analytics_Project\project\models\speech_pipeline"
TEXT_PATH   = r"C:\Speech_Analytics_Project\project\models\text_pipeline"
RESULTS_PATH = r"C:\Speech_Analytics_Project\project\Results\plots"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ─── LOAD SAVED EMBEDDINGS ────────────────────────────────
print("Loading speech embeddings...")
speech_X = np.load(os.path.join(SPEECH_PATH, 'hubert_X.npy'))
speech_y = np.load(os.path.join(SPEECH_PATH, 'hubert_y.npy'), allow_pickle=True)

print("Loading text embeddings...")
text_X = np.load(os.path.join(TEXT_PATH, 'bert_X.npy'))
text_y = np.load(os.path.join(TEXT_PATH, 'bert_y.npy'), allow_pickle=True)

print(f"Speech embeddings shape: {speech_X.shape}")
print(f"Text embeddings shape:   {text_X.shape}")

# ─── FUSION — CONCATENATE SPEECH + TEXT ───────────────────
# Both have same number of samples (2798)
# Speech: (2798, 768) + Text: (2798, 768) = (2798, 1536)
X_fused = np.concatenate([speech_X, text_X], axis=1)
print(f"Fused shape: {X_fused.shape}")

# Use speech labels (same as text labels)
labels = speech_y

# ─── ENCODE LABELS ────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_cat = to_categorical(y_encoded)
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")

with open('label_encoder_fusion.pkl', 'wb') as f:
    pickle.dump(le, f)

# ─── TRAIN/TEST SPLIT ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

np.save('X_test_fusion.npy', X_test)
np.save('y_test_fusion.npy', y_test)

# ─── BUILD FUSION MODEL ───────────────────────────────────
# Input is 1536 (768 speech + 768 text concatenated)
model = Sequential([
    Dense(512, activation='relu', input_shape=(1536,)),
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
print("\nTraining Fusion model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=80,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save('fusion_emotion_model.h5')
print("Model saved as fusion_emotion_model.h5")

# ─── EVALUATE ─────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFusion Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ─── PLOT 1: Training Curves ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Fusion Model — Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Fusion Model — Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'fusion_train_curves.png'), dpi=150)
plt.close()
print("Saved: fusion_train_curves.png")

# ─── PLOT 2: Confusion Matrix ─────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Oranges')
plt.title('Fusion Model — Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'fusion_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: fusion_confusion_matrix.png")

# ─── PLOT 3: Final Accuracy Comparison Table ──────────────
models = ['Speech Only\n(HuBERT)', 'Text Only\n(BERT)', 'Multimodal\n(Fusion)']
accuracies = [99.64, 100.00, acc*100]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#4C72B0', '#55A868', '#C44E52'], width=0.4)
plt.ylim([90, 101])
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison — All 3 Pipelines')
for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'model_comparison.png'), dpi=150)
plt.close()
print("Saved: model_comparison.png")

print("\n✅ Fusion pipeline training complete!")
print(f"\n{'='*40}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*40}")
print(f"  Speech Only  : 99.64%")
print(f"  Text Only    : 100.00%")
print(f"  Multimodal   : {acc*100:.2f}%")
print(f"{'='*40}")
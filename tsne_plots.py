import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import os

# ─── PATHS ────────────────────────────────────────────────
SPEECH_PATH = r"C:\Speech_Analytics_Project\project\models\speech_pipeline"
TEXT_PATH   = r"C:\Speech_Analytics_Project\project\models\text_pipeline"
FUSION_PATH = r"C:\Speech_Analytics_Project\project\models\fusion_pipeline"
RESULTS_PATH = r"C:\Speech_Analytics_Project\project\Results\plots"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ─── COLORS FOR 7 EMOTIONS ────────────────────────────────
COLORS = {
    'angry':    '#FF4444',
    'disgust':  '#8B008B',
    'fear':     '#FF8C00',
    'happy':    '#FFD700',
    'neutral':  '#4682B4',
    'sad':      '#1E90FF',
    'surprise': '#32CD32'
}

# ─── LOAD DATA ────────────────────────────────────────────
print("Loading embeddings...")

# Speech embeddings (HuBERT)
speech_X = np.load(os.path.join(SPEECH_PATH, 'hubert_X.npy'))
speech_y = np.load(os.path.join(SPEECH_PATH, 'hubert_y.npy'), allow_pickle=True)

# Text embeddings (BERT)
text_X = np.load(os.path.join(TEXT_PATH, 'bert_X.npy'))
text_y = np.load(os.path.join(TEXT_PATH, 'bert_y.npy'), allow_pickle=True)

# Fusion embeddings
fusion_X = np.concatenate([speech_X, text_X], axis=1)
fusion_y = speech_y

print(f"Speech embeddings: {speech_X.shape}")
print(f"Text embeddings:   {text_X.shape}")
print(f"Fusion embeddings: {fusion_X.shape}")

# ─── FUNCTION TO PLOT t-SNE ───────────────────────────────
def plot_tsne(X, y, title, filename):
    print(f"\nGenerating t-SNE for: {title}")
    print("This may take 2-3 minutes...")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 8))

    emotions = np.unique(y)
    for emotion in emotions:
        mask = y == emotion
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=COLORS[emotion],
            label=emotion.capitalize(),
            alpha=0.6,
            s=30
        )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1),
               loc='upper left', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# ─── GENERATE ALL 3 PLOTS ─────────────────────────────────

# Plot 1 — Speech (Temporal Modelling block)
plot_tsne(
    speech_X, speech_y,
    't-SNE Visualization — Speech (HuBERT Temporal Modelling Block)',
    'tsne_speech_temporal.png'
)

# Plot 2 — Text (Contextual Modelling block)
plot_tsne(
    text_X, text_y,
    't-SNE Visualization — Text (BERT Contextual Modelling Block)',
    'tsne_text_contextual.png'
)

# Plot 3 — Fusion block
plot_tsne(
    fusion_X, fusion_y,
    't-SNE Visualization — Multimodal Fusion Block',
    'tsne_fusion.png'
)

print("\n✅ All 3 t-SNE plots saved in Results/plots/")
print("Files:")
print("  → tsne_speech_temporal.png")
print("  → tsne_text_contextual.png")
print("  → tsne_fusion.png")
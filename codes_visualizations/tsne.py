import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# === Load Data ===
real_embeddings = np.load("embeds/real_embeddings.npy")
ai_embeddings = np.load("embeds/ai_embeddings.npy")

# Also load the filenames (ensure same order as embeddings!)
with open("embeds/real_filenames.txt") as f:
    real_files = [line.strip() for line in f]
with open("embeds/ai_filenames.txt") as f:
    ai_files = [line.strip() for line in f]

# === Combine ===
X = np.vstack([real_embeddings, ai_embeddings])
all_files = real_files + ai_files

# === Labels for Plotting ===
labels = []
for fname in all_files:
    if "VITS" in fname:
        labels.append("VITS")
    elif "Tacotron2" in fname:
        labels.append("Tacotron2")
    elif "FastSpeech2" in fname:
        labels.append("FastSpeech2")
    elif "common_voice" in fname:  # real audio
        labels.append("Real")
    else:
        labels.append("Unknown")

# === t-SNE ===
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X)

# === Plot ===
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1],
    hue=labels,
    palette={
        "Real": "green",
        "VITS": "blue",
        "Tacotron2": "purple",
        "FastSpeech2": "orange",
        "Unknown": "black"
    },
    alpha=0.7,
    edgecolor=None,
    s=30
)
plt.title("t-SNE: Wav2Vec Embeddings by TTS Model")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("tsne_by_tts_model.png", dpi=300)


import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
from tqdm import tqdm
import numpy as np

# Load Wav2Vec2 base model (you can use a larger one too)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Paths to your resampled datasets
real_dir = "/home/sashank/Desktop/project/wav_2vec/real_16"
ai_dir = "/home/sashank/Desktop/project/wav_2vec/ai_16"
embedding_save_dir = "/home/sashank/Desktop/project/wav_2vec/embeds"
os.makedirs(embedding_save_dir, exist_ok=True)

def extract_embeddings(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze()  # Make sure it's 1D
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # shape: (1, time, feature_dim)
    
    # Option 1: Mean pooling over time
    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def process_folder(folder_path, label):
    all_embeddings = []
    file_names = []
    
    for fname in tqdm(sorted(os.listdir(folder_path))):
        if fname.endswith(".wav"):
            fpath = os.path.join(folder_path, fname)
            emb = extract_embeddings(fpath)
            all_embeddings.append(emb)
            file_names.append((fname, label))

    return np.array(all_embeddings), file_names

# Process both classes
real_embeddings, real_info = process_folder(real_dir, 0)
ai_embeddings, ai_info = process_folder(ai_dir, 1)

# Save embeddings and labels
np.save(os.path.join(embedding_save_dir, "real_embeddings.npy"), real_embeddings)
np.save(os.path.join(embedding_save_dir, "ai_embeddings.npy"), ai_embeddings)

with open(os.path.join(embedding_save_dir, "labels.txt"), "w") as f:
    for fname, label in real_info + ai_info:
        f.write(f"{fname}\t{label}\n")


# Part 1: Feature Extraction and Dataset Creation

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 16000  # Standardize sample rate
MAX_AUDIO_LENGTH = 12  # Maximum audio length in seconds
FEATURE_TYPE = 'mfcc'  # 'mfcc' or 'mel'
BATCH_SIZE = 5

class AudioFeatureExtractor:
    def __init__(self, feature_type='mfcc', sample_rate=16000, max_length=5):
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_samples = max_length * sample_rate
        
        if feature_type == 'mfcc':
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, 
                n_mfcc=40,
                log_mels=True
            )
        elif feature_type == 'mel':
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128
            )
        else:
            raise ValueError("Feature type must be 'mfcc' or 'mel'")
            
    def extract(self, audio_path):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate to fixed length
        if waveform.shape[1] < self.max_samples:
            # Pad
            padding = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Truncate
            waveform = waveform[:, :self.max_samples]
        
        # Extract features
        features = self.transform(waveform)
        
        # Apply log to mel spectrograms for better scaling
        if self.feature_type == 'mel':
            features = torch.log(features + 1e-9)  # Add small constant to avoid log(0)
            
        return features

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Add channel dimension for CNN (B, C, H, W)
        feature = self.features[idx]
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)    	
        label = self.labels[idx]
        return feature, label

def prepare_dataset(real_folder, ai_folder, feature_type='mfcc', test_size=0.2):
    extractor = AudioFeatureExtractor(feature_type=feature_type)
    
    # Process real and AI audio files
    real_features, real_labels = [], []
    ai_features, ai_labels = [], []
    
    # Process real audio
    print(f"Processing real audio from {real_folder}...")
    for fname in tqdm(sorted(os.listdir(real_folder))):
        if fname.endswith(('.wav', '.mp3')):
            path = os.path.join(real_folder, fname)
            try:
                feat = extractor.extract(path)
                real_features.append(feat)
                real_labels.append(0)  # 0 for real
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    # Process AI audio
    print(f"Processing AI audio from {ai_folder}...")
    for fname in tqdm(sorted(os.listdir(ai_folder))):
        if fname.endswith(('.wav', '.mp3')):
            path = os.path.join(ai_folder, fname)
            try:
                feat = extractor.extract(path)
                ai_features.append(feat)
                ai_labels.append(1)  # 1 for AI
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    # Combine datasets
    all_features = real_features + ai_features
    all_labels = real_labels + ai_labels
    
    # Convert to tensors
    features_tensor = torch.stack(all_features)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features_tensor, labels_tensor, 
        test_size=test_size, 
        random_state=42, 
        stratify=labels_tensor
    )
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, X_train.shape[2:]

if __name__ == "__main__":
    # Example usage
    real_audio_folder = "/home/sashank/Desktop/project/wav_2vec/real_16"
    ai_audio_folder = "/home/sashank/Desktop/project/wav_2vec/ai_16"
    
    train_loader, test_loader, input_shape = prepare_dataset(
        real_audio_folder,
        ai_audio_folder,
        feature_type='mfcc'
    )
    
    print(f"Feature shape: {input_shape}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

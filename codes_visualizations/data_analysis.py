import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from part1_fe import AudioFeatureExtractor, prepare_dataset

def analyze_dataset_split(real_folder, ai_folder, feature_type='mfcc', output_dir="data_analysis"):
    """
    Analyze the dataset and visualize the train/test split to check for potential issues
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the data loaders
    train_loader, test_loader, input_shape = prepare_dataset(
        real_folder,
        ai_folder,
        feature_type=feature_type
    )
    
    # Extract features and labels from train and test sets
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    print("Extracting train features...")
    for features, labels in train_loader:
        for i in range(features.size(0)):
            # Flatten the features for dimensionality reduction
            flat_feature = features[i].view(-1).numpy()
            train_features.append(flat_feature)
            train_labels.append(labels[i].item())
    
    print("Extracting test features...")
    for features, labels in test_loader:
        for i in range(features.size(0)):
            flat_feature = features[i].view(-1).numpy()
            test_features.append(flat_feature)
            test_labels.append(labels[i].item())
    
    # Convert to numpy arrays
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    print(f"Train set: {len(train_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA...")
    pca = PCA(n_components=50)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    
    # Apply t-SNE for visualization
    print("Applying t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Combine train and test for t-SNE
    combined_pca = np.vstack((train_pca, test_pca))
    combined_labels = np.concatenate((train_labels, test_labels))
    combined_is_train = np.concatenate((np.ones(len(train_pca)), np.zeros(len(test_pca))))
    
    # Fit t-SNE on the combined data
    tsne_results = tsne.fit_transform(combined_pca)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Plot by class (Real vs AI)
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_labels, 
                         cmap=plt.cm.get_cmap('viridis', 2), alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label="Class (0=Real, 1=AI)")
    plt.title('t-SNE Visualization by Class (Real vs AI)')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    
    # Plot by split (Train vs Test)
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=combined_is_train, 
                         cmap=plt.cm.get_cmap('coolwarm', 2), alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label="Split (0=Test, 1=Train)")
    plt.title('t-SNE Visualization by Split (Train vs Test)')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze class distribution
    train_class_dist = np.bincount(train_labels) / len(train_labels)
    test_class_dist = np.bincount(test_labels) / len(test_labels)
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.array([0, 1])
    plt.bar(x - width/2, train_class_dist, width, label='Train')
    plt.bar(x + width/2, test_class_dist, width, label='Test')
    plt.xticks(x, ['Real', 'AI'])
    plt.ylabel('Proportion')
    plt.title('Class Distribution in Train and Test Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute and analyze feature statistics
    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)
    test_mean = np.mean(test_features, axis=0)
    test_std = np.std(test_features, axis=0)
    
    # Plot feature means comparison (first 100 features)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_mean[:100], label='Train')
    plt.plot(test_mean[:100], label='Test')
    plt.title('Feature Means (first 100 features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature std comparison
    plt.subplot(1, 2, 2)
    plt.plot(train_std[:100], label='Train')
    plt.plot(test_std[:100], label='Test')
    plt.title('Feature Standard Deviations (first 100 features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis results saved to {output_dir}")
    
    # Return statistical measures of similarity between train and test
    mean_diff = np.mean(np.abs(train_mean - test_mean))
    std_diff = np.mean(np.abs(train_std - test_std))
    
    print(f"Mean absolute difference between train and test means: {mean_diff:.6f}")
    print(f"Mean absolute difference between train and test standard deviations: {std_diff:.6f}")
    
    # Compare class-specific statistics
    for class_id in [0, 1]:  # 0=Real, 1=AI
        train_class_mean = np.mean(train_features[train_labels == class_id], axis=0)
        test_class_mean = np.mean(test_features[test_labels == class_id], axis=0)
        class_mean_diff = np.mean(np.abs(train_class_mean - test_class_mean))
        print(f"Class {class_id} mean difference between train and test: {class_mean_diff:.6f}")

if __name__ == "__main__":
    # Set paths to your dataset folders
    real_audio_folder = "/home/sashank/Desktop/project/wav_2vec/real_16"
    ai_audio_folder = "/home/sashank/Desktop/project/wav_2vec/ai_16"
    output_dir = "/home/sashank/Desktop/project/wav_2vec/data_analysis"
    
    # Run the analysis
    analyze_dataset_split(real_audio_folder, ai_audio_folder, feature_type='mfcc', output_dir=output_dir)

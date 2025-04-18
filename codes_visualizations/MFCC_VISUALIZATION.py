import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from part1_fe import AudioFeatureExtractor

def visualize_mfccs(real_folder, ai_folder, num_samples=5, save_dir="mfcc_visualizations"):
    """
    Visualize MFCC features from real and AI audio samples
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(feature_type='mfcc')
    
    # Get random samples from each folder
    real_files = [f for f in os.listdir(real_folder) if f.endswith(('.wav', '.mp3'))]
    ai_files = [f for f in os.listdir(ai_folder) if f.endswith(('.wav', '.mp3'))]
    
    # Randomly select files
    np.random.seed(42)  # For reproducibility
    real_samples = np.random.choice(real_files, min(num_samples, len(real_files)), replace=False)
    ai_samples = np.random.choice(ai_files, min(num_samples, len(ai_files)), replace=False)
    
    # Process selected files
    for i, (real_file, ai_file) in enumerate(zip(real_samples, ai_samples)):
        real_path = os.path.join(real_folder, real_file)
        ai_path = os.path.join(ai_folder, ai_file)
        
        # Extract features
        try:
            real_mfcc = extractor.extract(real_path)
            ai_mfcc = extractor.extract(ai_path)
            
            # Create comparison plot
            plt.figure(figsize=(15, 10))
            
            # Plot real audio MFCC
            plt.subplot(2, 1, 1)
            plt.title(f"Real Audio MFCC: {real_file}")
            plt.imshow(real_mfcc.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time Frames')
            plt.ylabel('MFCC Coefficients')
            
            # Plot AI audio MFCC
            plt.subplot(2, 1, 2)
            plt.title(f"AI Audio MFCC: {ai_file}")
            plt.imshow(ai_mfcc.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time Frames')
            plt.ylabel('MFCC Coefficients')
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(save_dir, f"mfcc_comparison_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also save the difference between the two
            plt.figure(figsize=(12, 6))
            plt.title(f"Difference between Real and AI MFCCs (Real - AI)")
            diff = real_mfcc.squeeze(0).numpy() - ai_mfcc.squeeze(0).numpy()
            img = plt.imshow(diff, aspect='auto', origin='lower', cmap='coolwarm')
            plt.colorbar(img, format='%+2.0f dB')
            plt.xlabel('Time Frames')
            plt.ylabel('MFCC Coefficients')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"mfcc_difference_{i+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing {real_file} or {ai_file}: {e}")
    
    # Create an additional visualization showing average differences
    try:
        # Process more samples for average
        num_avg_samples = min(50, len(real_files), len(ai_files))
        real_avg_samples = np.random.choice(real_files, num_avg_samples, replace=False)
        ai_avg_samples = np.random.choice(ai_files, num_avg_samples, replace=False)
        
        real_mfccs = []
        ai_mfccs = []
        
        # Extract features from multiple samples
        print("Extracting features for average analysis...")
        for real_file in tqdm(real_avg_samples):
            try:
                real_path = os.path.join(real_folder, real_file)
                real_mfcc = extractor.extract(real_path)
                real_mfccs.append(real_mfcc.squeeze(0).numpy())
            except Exception as e:
                print(f"Error processing {real_file}: {e}")
                
        for ai_file in tqdm(ai_avg_samples):
            try:
                ai_path = os.path.join(ai_folder, ai_file)
                ai_mfcc = extractor.extract(ai_path)
                ai_mfccs.append(ai_mfcc.squeeze(0).numpy())
            except Exception as e:
                print(f"Error processing {ai_file}: {e}")
        
        # Calculate averages
        real_avg = np.mean(np.array(real_mfccs), axis=0)
        ai_avg = np.mean(np.array(ai_mfccs), axis=0)
        avg_diff = real_avg - ai_avg
        
        # Plot average MFCCs
        plt.figure(figsize=(15, 15))
        
        plt.subplot(3, 1, 1)
        plt.title(f"Average Real Audio MFCC (n={len(real_mfccs)})")
        plt.imshow(real_avg, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time Frames')
        plt.ylabel('MFCC Coefficients')
        
        plt.subplot(3, 1, 2)
        plt.title(f"Average AI Audio MFCC (n={len(ai_mfccs)})")
        plt.imshow(ai_avg, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time Frames')
        plt.ylabel('MFCC Coefficients')
        
        plt.subplot(3, 1, 3)
        plt.title("Average Difference (Real - AI)")
        img = plt.imshow(avg_diff, aspect='auto', origin='lower', cmap='coolwarm')
        plt.colorbar(img, format='%+2.0f dB')
        plt.xlabel('Time Frames')
        plt.ylabel('MFCC Coefficients')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "average_mfcc_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also plot the standard deviation of the differences to highlight areas of high variability
        real_std = np.std(np.array(real_mfccs), axis=0)
        ai_std = np.std(np.array(ai_mfccs), axis=0)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.title(f"Standard Deviation of Real Audio MFCCs (n={len(real_mfccs)})")
        plt.imshow(real_std, aspect='auto', origin='lower', cmap='plasma')
        plt.colorbar()
        plt.xlabel('Time Frames')
        plt.ylabel('MFCC Coefficients')
        
        plt.subplot(2, 1, 2)
        plt.title(f"Standard Deviation of AI Audio MFCCs (n={len(ai_mfccs)})")
        plt.imshow(ai_std, aspect='auto', origin='lower', cmap='plasma')
        plt.colorbar()
        plt.xlabel('Time Frames')
        plt.ylabel('MFCC Coefficients')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mfcc_standard_deviations.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating average visualizations: {e}")
    
    print(f"Visualizations saved to {save_dir}")

if __name__ == "__main__":
    # Set paths to your dataset folders
    real_audio_folder = "/home/sashank/Desktop/project/wav_2vec/real_16"
    ai_audio_folder = "/home/sashank/Desktop/project/wav_2vec/ai_16"
    save_dir = "/home/sashank/Desktop/project/wav_2vec/mfcc_visualizations"
    
    # Generate visualizations
    visualize_mfccs(real_audio_folder, ai_audio_folder, num_samples=5, save_dir=save_dir)

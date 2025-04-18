import os
import torch
import json
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from TTS.api import TTS
import pandas as pd

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Define TTS models to use
TTS_MODELS = {
    "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
    "FastSpeech2": "tts_models/en/ljspeech/fast_pitch",
    "VITS": "tts_models/en/ljspeech/vits",
    "YourTTS": "tts_models/multilingual/multi-dataset/your_tts"
}

# Configuration
real_audio_dir = Path("/home/sashank/Desktop/project/tt")  # CHANGE THIS
artificial_text_file = Path("/home/sashank/Desktop/project/50.txt")  # CHANGE THIS
output_dir = Path("synthetic_speech")
batch_size = 10  # Process files in batches to save memory

# Create output directory
output_dir.mkdir(exist_ok=True, parents=True)

# Unified metadata file
metadata = {
    "dataset_info": {
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "real_audio_dir": str(real_audio_dir),
        "artificial_text_file": str(artificial_text_file),
        "output_dir": str(output_dir),
        "random_seed": RANDOM_SEED,
        "tts_models_used": list(TTS_MODELS.keys())
    },
    "samples": []
}

# Statistics counters
stats = {
    "total_real_samples": 0,
    "successful_generations": 0,
    "failed_generations": 0,
    "model_usage": {model: 0 for model in TTS_MODELS},
    "model_success": {model: 0 for model in TTS_MODELS},
    "model_failure": {model: 0 for model in TTS_MODELS}
}

# Error log file
error_log_path = output_dir / "error_log.txt"
with open(error_log_path, "w", encoding="utf-8") as error_log:
    error_log.write(f"Error log created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    error_log.write("=" * 80 + "\n\n")

# Load models
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tts_instances = {}
for model_name, model_path in TTS_MODELS.items():
    try:
        tts_instances[model_name] = TTS(model_path).to(device)
        print(f"Loaded {model_name} successfully!")
    except Exception as e:
        error_msg = f"Error loading {model_name}: {e}"
        print(error_msg)
        with open(error_log_path, "a", encoding="utf-8") as error_log:
            error_log.write(error_msg + "\n")

# Check if we have any models loaded
if not tts_instances:
    print("No TTS models could be loaded. Exiting.")
    exit(1)

# Input real audio files
real_audio_files = list(real_audio_dir.glob("*.mp3"))  # Adjust extension as needed
stats["total_real_samples"] = len(real_audio_files)

# Load artificial text file
with open(artificial_text_file, "r", encoding="utf-8") as f:
    artificial_texts = [line.strip() for line in f.readlines()]

# Ensure the number of texts matches the number of audio files
if len(artificial_texts) != len(real_audio_files):
    print(f"Warning: Number of texts ({len(artificial_texts)}) does not match number of audio files ({len(real_audio_files)}).")
    if len(artificial_texts) < len(real_audio_files):
        print("There are fewer texts than audio files. Some audio files will be skipped.")
        real_audio_files = real_audio_files[:len(artificial_texts)]
    else:
        print("There are more texts than audio files. Some texts will be unused.")
        artificial_texts = artificial_texts[:len(real_audio_files)]

# Process in batches with progress tracking
num_batches = (len(real_audio_files) + batch_size - 1) // batch_size
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(real_audio_files))
    batch_files = real_audio_files[start_idx:end_idx]
    batch_texts = artificial_texts[start_idx:end_idx]
    
    print(f"\nProcessing batch {batch_idx+1}/{num_batches} ({len(batch_files)} files)")
    
    # Process each real audio sample in this batch
    for audio_path, text in tqdm(zip(batch_files, batch_texts), total=len(batch_files), desc=f"Batch {batch_idx+1}"):
        audio_name = audio_path.stem
        
        # Randomly select one TTS model
        model_name = random.choice(list(tts_instances.keys()))
        tts = tts_instances[model_name]
        stats["model_usage"][model_name] += 1
        
        output_path = output_dir / f"{audio_name}_{model_name}.mp3"
        
        sample_metadata = {
            "original": str(audio_path),
            "text": text,
            "model_used": model_name,
            "output_path": str(output_path)
        }
        
        try:
            # Generate synthetic speech
            tts.tts_to_file(
                text=text,
                file_path=str(output_path)
            )
            
            sample_metadata["status"] = "success"
            stats["successful_generations"] += 1
            stats["model_success"][model_name] += 1
            
        except Exception as e:
            error_msg = f"Error generating {model_name} speech for {audio_path}: {e}"
            with open(error_log_path, "a", encoding="utf-8") as error_log:
                error_log.write(error_msg + "\n")
            
            sample_metadata["status"] = "failed"
            sample_metadata["error"] = str(e)
            stats["failed_generations"] += 1
            stats["model_failure"][model_name] += 1
        
        metadata["samples"].append(sample_metadata)
        
    # Save metadata after each batch
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

# Create CSV summary for easier analysis
samples_df = pd.DataFrame(metadata["samples"])
csv_path = output_dir / "dataset_summary.csv"
samples_df.to_csv(csv_path, index=False)

# Calculate final statistics
stats["completion_percentage"] = (stats["successful_generations"] / stats["total_real_samples"]) * 100 if stats["total_real_samples"] > 0 else 0
for model in TTS_MODELS:
    if stats["model_usage"][model] > 0:
        stats["model_success_rate"] = {model: (stats["model_success"][model] / stats["model_usage"][model]) * 100 for model in TTS_MODELS}
    else:
        stats["model_success_rate"] = {model: 0 for model in TTS_MODELS}

# Add statistics to metadata
metadata["statistics"] = stats

# Save final metadata
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

# Print final statistics
print("\n" + "="*50)
print("Synthetic Speech Generation Completed!")
print("="*50)
print(f"Total real samples: {stats['total_real_samples']}")
print(f"Successful generations: {stats['successful_generations']} ({stats['completion_percentage']:.2f}%)")
print(f"Failed generations: {stats['failed_generations']}")
print("\nModel usage statistics:")
for model in TTS_MODELS:
    success_rate = stats["model_success_rate"][model]
    print(f"  {model}: Used {stats['model_usage'][model]} times, {stats['model_success'][model]} successful ({success_rate:.2f}% success rate)")

print(f"\nMetadata saved to: {metadata_path}")
print(f"CSV summary saved to: {csv_path}")
print(f"Error log saved to: {error_log_path}")
print("="*50)

import os
import librosa
from pydub import AudioSegment
from tqdm import tqdm

def normalize_audio(input_dir: str, output_dir: str, target_sr: int = 24000):
    """
    Normalize audio to target sample rate (22.05kHz or 24kHz)
    Args:
        input_dir: Folder with input audio files
        output_dir: Folder to save normalized files
        target_sr: Target sample rate (22050 or 24000)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.lower().endswith(('.wav', '.mp3')):
            continue
            
        try:
            # Load audio
            input_path = os.path.join(input_dir, fname)
            audio = AudioSegment.from_file(input_path)
            
            # Convert to target sample rate
            audio = audio.set_frame_rate(target_sr)
            
            # Normalize loudness (optional)
            audio = audio.normalize()
            
            # Export as WAV
            output_path = os.path.join(output_dir, fname)
            audio.export(output_path, format="wav", parameters=["-ac", "1"])  # Force mono
            
            print(f"Processed: {fname} -> {target_sr}Hz")
            
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")

if __name__ == "__main__":
    normalize_audio(
        input_dir="raw_audio",       # Change to your input folder
        output_dir="normalized_audio",
        target_sr=24000             # 22050 or 24000
    )

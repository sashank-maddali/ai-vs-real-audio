from pydub import AudioSegment
import os

input_dir = "/home/sashank/Desktop/project/wav_2vec/ai"             # Your folder with AI-generated .wav files
output_dir = "/home/sashank/Desktop/project/wav_2vec/ai_16"         # Output folder for 16kHz files

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        wav_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        audio = AudioSegment.from_wav(wav_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        print(f"Resampled: {filename}")


import torch
from faster_whisper import WhisperModel
import os

class STTEngine:
    def __init__(self, model_size = "./whisper_model"):
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN")  
        if not hf_token:
            print("HF_TOKEN not found in .env")

        # Use float16 for GPU, int8 for CPU
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        
        print(f"Loading Whisper model '{model_size}' on {self.device} ({self.compute_type})...")
        self.model = WhisperModel(
            model_size, 
            device=self.device, 
            compute_type=self.compute_type, 
            cpu_threads=6
        )
    
    def transcribe(self, audio_file):
        # We return a generator to allow UI updates
        segments, info = self.model.transcribe(audio_file, beam_size=1)
        return segments, info
    
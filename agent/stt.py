from faster_whisper import WhisperModel
import os 


class STTEngine:
    def __init__(self, model_size = "./whisper_model"):
        self.model = WhisperModel(model_size, device= "cpu",compute_type= "int8")
        hf_token = os.getenv("HF_TOKEN")  
        if not hf_token:
            print("HF_TOKEN not found in .env")
    
    def transcribe(self, audio_file):
        segment, _ = self.model.transcribe(audio_file, beam_size=1)
        return " ".join([s.text for s in segment]).strip()
    
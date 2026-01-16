from faster_whisper import WhisperModel
import os


def load_model(model_size: str, hw: str, ctype: str ):
    model_path = os.path.join("models", "faster-whisper-"+model_size)
    print(model_path)
    model = WhisperModel(model_path, device=hw, compute_type=ctype)
    return model

def transcribe(model: WhisperModel, file_path: str):
    segments, info = model.transcribe(
        file_path, 
        language = "en", 
        beam_size = 5
    )
    return segments, info


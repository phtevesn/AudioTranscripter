from faster_whisper import WhisperModel

def load_model(model_size: str, hw: str, ctype: str ):
    model = WhisperModel(model_size, device=hw, compute_type=ctype)
    return model

def transcribe(model: WhisperModel, file_path: str):
    segments, info = model.transcribe(
        file_path, 
        language = "en", 
        beam_size = 5
    )
    return segments, info


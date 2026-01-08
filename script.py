from faster_whisper import WhisperModel

#model_size = "large-v3"
#model = WhisperModel(model_size, device="cpu", compute_type="int8")

def create_model(model_size: str, device_in: str, compute_type_in: str):
    return WhisperModel(model_size, device_in, compute_type_in)

def generate_text(wav_path: str, model: WhisperModel):
    segments, info = model.transcribe("test.wav", beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    

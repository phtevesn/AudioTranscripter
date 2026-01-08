from faster_whisper import WhisperModel
from datetime import datetime

start_time = datetime.now()
print(start_time)
model_size = "small.en"
'''
tiny
base
small
medium
large-v2
large-v3
tiny.en
base.en
small.en
medium.en
'''
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("one.wav", beam_size=5)

output_path = "transcript.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(
        "Detected language '%s' with probability %.4f\n\n"
        % (info.language, info.language_probability)
    )

    for segment in segments:
        f.write(
            "[%.2fs -> %.2fs] %s\n"
            % (segment.start, segment.end, segment.text)
        )

print(f"Saved transcript to {output_path}")


end_time = datetime.now
print(end_time)
print(f"Diff: {end_time - start_time}")
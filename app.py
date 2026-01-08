import os
import threading
from tkinter import Tk, Label, StringVar
from tkinterdnd2 import DND_FILES, TkinterDnD

from faster_whisper import WhisperModel


# ---- Load Whisper model once ----
model = WhisperModel(
    "base",            # change if you want
    device="cpu",
    compute_type="int8"
)


def transcribe(file_path, status_var):
    status_var.set("Transcribing...")
    
    segments, info = model.transcribe(
        file_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )

    out_file = os.path.splitext(file_path)[0] + ".txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}\n")

    status_var.set(f"Done! Saved: {out_file}")


def on_drop(event):
    # Windows drag/drop gives braces sometimes
    file_path = event.data.strip("{}")

    if not file_path.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        status.set("Unsupported file type")
        return

    # Run transcription in background thread
    threading.Thread(
        target=transcribe,
        args=(file_path, status),
        daemon=True
    ).start()


# ---- UI ----
app = TkinterDnD.Tk()
app.title("Whisper Audio Transcriber")
app.geometry("500x200")

status = StringVar()
status.set("Drag & drop an audio file here")

label = Label(
    app,
    textvariable=status,
    relief="ridge",
    width=60,
    height=6
)
label.pack(padx=20, pady=40)

label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", on_drop)

app.mainloop()

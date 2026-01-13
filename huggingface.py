# download_models.py
from huggingface_hub import snapshot_download
from pathlib import Path

SIZES = ["tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3"
]  # pick what you want to ship: tiny/base/small/medium/large-v3

out_root = Path("models")
out_root.mkdir(exist_ok=True)

for size in SIZES:
    repo_id = f"Systran/faster-whisper-{size}"
    local_dir = out_root / f"faster-whisper-{size}"

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # IMPORTANT: makes it portable on Windows
    )

    print("Downloaded:", repo_id, "->", local_dir)

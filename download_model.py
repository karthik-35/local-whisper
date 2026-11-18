"""Download model files only. Never accepts or opens a recording or transcript."""
import json
import os
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
repo = "mlx-community/whisper-large-v3-turbo"
metadata = root / "model-cache.json"
previous = json.loads(metadata.read_text()) if metadata.exists() else {}
path = snapshot_download(repo_id=repo, revision=previous.get("revision"),
                         allow_patterns=["config.json", "weights.safetensors", "README.md"])
metadata.write_text(json.dumps({"repo": repo, "path": path, "revision": Path(path).name}, indent=2) + "\n")
print("Model cached locally:", path)

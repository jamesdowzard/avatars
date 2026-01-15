"""Configuration and paths for avatar generation."""
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
EXECUTIVES_DIR = ROOT / "executives"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "photos").mkdir(exist_ok=True)
(DATA_DIR / "stylized").mkdir(exist_ok=True)
(DATA_DIR / "audio").mkdir(exist_ok=True)
(DATA_DIR / "output").mkdir(exist_ok=True)


def get_executive(exec_id: str) -> dict:
    """Load executive metadata."""
    meta_path = EXECUTIVES_DIR / exec_id / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"Executive '{exec_id}' not found")
    return json.loads(meta_path.read_text())


def get_executive_photo(exec_id: str, photo_key: str = "primary") -> Path:
    """Get path to executive's photo."""
    meta = get_executive(exec_id)
    photo_name = meta["photos"].get(photo_key)
    if not photo_name:
        raise ValueError(f"Photo '{photo_key}' not found for {exec_id}")
    return EXECUTIVES_DIR / exec_id / photo_name


def list_executives() -> list[str]:
    """List all available executive IDs."""
    return [
        d.name for d in EXECUTIVES_DIR.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ]

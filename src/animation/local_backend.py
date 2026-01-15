"""Local animation using SadTalker or LivePortrait (requires GPU)."""
from pathlib import Path
from rich.console import Console

console = Console()


def animate_local(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    model: str = "sadtalker",
) -> Path:
    """
    Animate an image with audio using local models.

    Requires cloning and setting up SadTalker or LivePortrait.

    Args:
        image_path: Path to source image
        audio_path: Path to audio file (wav/mp3)
        output_path: Path to save output video
        model: Model to use ('sadtalker' or 'liveportrait')

    Returns:
        Path to generated video
    """
    # TODO: Implement local model support
    # This requires:
    # 1. Clone SadTalker: git clone https://github.com/OpenTalker/SadTalker
    # 2. Download model weights
    # 3. Set up the inference pipeline

    raise NotImplementedError(
        "Local animation not yet implemented. Use Replicate backend for now:\n"
        "  export REPLICATE_API_TOKEN=your_token\n"
        "  avatars animate --backend replicate ...\n\n"
        "To set up local:\n"
        "  git clone https://github.com/OpenTalker/SadTalker models/SadTalker\n"
        "  cd models/SadTalker && bash scripts/download_models.sh"
    )

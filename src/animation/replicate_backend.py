"""Animation using Replicate API (SadTalker, LivePortrait)."""
import os
import httpx
import replicate
from pathlib import Path
from rich.console import Console

console = Console()

# Available models on Replicate
MODELS = {
    "sadtalker": "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    "liveportrait": "fofr/live-portrait:067dd98cc1a99abb36c224bbb2b573153712a72bea24cb5966be065023c5edd7",
}


def animate_with_replicate(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    model: str = "liveportrait",
) -> Path:
    """
    Animate an image with audio using Replicate API.

    Args:
        image_path: Path to source image
        audio_path: Path to audio file (wav/mp3)
        output_path: Path to save output video
        model: Model to use ('sadtalker' or 'liveportrait')

    Returns:
        Path to generated video
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODELS.keys())}")

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set. Get one at https://replicate.com/account/api-tokens"
        )

    console.print(f"[blue]Animating with {model}...[/blue]")

    # Upload files and run model
    with open(image_path, "rb") as img, open(audio_path, "rb") as aud:
        if model == "sadtalker":
            output = replicate.run(
                MODELS[model],
                input={
                    "source_image": img,
                    "driven_audio": aud,
                    "still_mode": False,
                    "preprocess": "crop",
                    "expression_scale": 1.0,
                }
            )
        else:  # liveportrait
            output = replicate.run(
                MODELS[model],
                input={
                    "face_image": img,
                    "driving_audio": aud,
                }
            )

    # Download result
    video_url = output if isinstance(output, str) else output[0] if output else None
    if not video_url:
        raise RuntimeError("No output from model")

    console.print(f"[green]Downloading result...[/green]")
    response = httpx.get(video_url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    console.print(f"[green]Saved to {output_path}[/green]")
    return output_path

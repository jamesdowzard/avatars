"""Style transfer using Replicate API."""
import os
import httpx
import replicate
from pathlib import Path
from rich.console import Console

console = Console()

# Style presets using different models/prompts
STYLE_PRESETS = {
    "pixar": {
        "model": "tencentarc/photomaker:ddfc2b08d209f9fa8c1uj52b42c8c10849a26e2b0cf3a5ce27ea3ec8d5ee5d",
        "prompt": "pixar style 3D animated character, disney pixar, friendly corporate executive, professional, clean background",
        "negative": "realistic, photograph, blurry, low quality",
    },
    "illustration": {
        "model": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "prompt": "professional corporate illustration, digital art style, clean lines, modern flat design, executive portrait",
        "negative": "photograph, realistic, 3d render, blurry",
    },
    "cartoon": {
        "model": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "prompt": "cartoon style portrait, clean vector art, corporate professional, friendly expression",
        "negative": "realistic, photograph, 3d, complex background",
    },
}


def stylize_with_replicate(
    image_path: Path,
    output_path: Path,
    style: str = "illustration",
    strength: float = 0.65,
) -> Path:
    """
    Convert a photo to a stylized avatar using Replicate.

    Args:
        image_path: Path to source photo
        output_path: Path to save stylized image
        style: Style preset ('pixar', 'illustration', 'cartoon')
        strength: How much to transform (0.0-1.0, higher = more stylized)

    Returns:
        Path to stylized image
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set. Get one at https://replicate.com/account/api-tokens"
        )

    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {style}. Choose from: {list(STYLE_PRESETS.keys())}")

    preset = STYLE_PRESETS[style]
    console.print(f"[blue]Stylizing with '{style}' style...[/blue]")

    with open(image_path, "rb") as img:
        # Use img2img for style transfer while maintaining likeness
        output = replicate.run(
            preset["model"],
            input={
                "image": img,
                "prompt": preset["prompt"],
                "negative_prompt": preset.get("negative", ""),
                "strength": strength,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }
        )

    # Get result URL
    result_url = output[0] if isinstance(output, list) else output
    if not result_url:
        raise RuntimeError("No output from style model")

    console.print(f"[green]Downloading stylized image...[/green]")
    response = httpx.get(result_url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    console.print(f"[green]Saved to {output_path}[/green]")
    return output_path

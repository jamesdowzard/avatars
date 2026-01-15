"""Text-to-speech using Replicate API."""
import os
import httpx
import replicate
from pathlib import Path
from rich.console import Console

console = Console()

# TTS models on Replicate
MODELS = {
    "bark": "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
    "xtts": "lucataco/xtts-v2:684bc3855b37866c0c65add2ff39c78f3dea3f4ff103a436465326e0f438d55e",
}


def generate_speech_replicate(
    text: str,
    output_path: Path,
    model: str = "bark",
    voice: str = "en_speaker_6",  # British male for bark
) -> Path:
    """
    Generate speech using Replicate API.

    Args:
        text: Text to speak
        output_path: Path to save audio file
        model: TTS model ('bark' or 'xtts')
        voice: Voice ID/preset

    Returns:
        Path to generated audio file
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set. Get one at https://replicate.com/account/api-tokens"
        )

    console.print(f"[blue]Generating speech with {model}...[/blue]")

    if model == "bark":
        output = replicate.run(
            MODELS[model],
            input={
                "prompt": text,
                "history_prompt": f"v2/{voice}",
                "text_temp": 0.7,
                "waveform_temp": 0.7,
            }
        )
    elif model == "xtts":
        output = replicate.run(
            MODELS[model],
            input={
                "text": text,
                "language": "en",
            }
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    # Download result
    audio_url = output.get("audio_out") if isinstance(output, dict) else output
    if not audio_url:
        raise RuntimeError("No audio output from model")

    console.print(f"[green]Downloading audio...[/green]")
    response = httpx.get(audio_url)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    console.print(f"[green]Saved to {output_path}[/green]")
    return output_path

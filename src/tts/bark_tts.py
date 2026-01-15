"""Text-to-speech using Bark (local)."""
from pathlib import Path
from rich.console import Console
import numpy as np

console = Console()

# Bark voice presets - these affect the speaker characteristics
VOICE_PRESETS = {
    "male_british": "v2/en_speaker_6",  # British male
    "male_american": "v2/en_speaker_0",  # American male
    "female_british": "v2/en_speaker_9",  # British female
    "female_american": "v2/en_speaker_1",  # American female
    "narrator": "v2/en_speaker_3",  # Neutral narrator
}


def generate_speech(
    text: str,
    output_path: Path,
    voice: str = "male_british",
) -> Path:
    """
    Generate speech from text using Bark.

    Args:
        text: Text to speak
        output_path: Path to save audio file (wav)
        voice: Voice preset name or bark voice ID

    Returns:
        Path to generated audio file
    """
    try:
        from bark import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav
    except ImportError:
        raise ImportError(
            "Bark not installed. Run: uv add git+https://github.com/suno-ai/bark.git"
        )

    console.print(f"[blue]Loading Bark models...[/blue]")
    preload_models()

    # Resolve voice preset
    voice_id = VOICE_PRESETS.get(voice, voice)

    console.print(f"[blue]Generating speech with voice '{voice}'...[/blue]")

    # Bark works best with shorter segments
    # For longer text, split into sentences
    sentences = _split_text(text)
    audio_arrays = []

    for i, sentence in enumerate(sentences):
        console.print(f"  [{i+1}/{len(sentences)}] {sentence[:50]}...")
        audio = generate_audio(sentence, history_prompt=voice_id)
        audio_arrays.append(audio)

    # Concatenate with small silence between sentences
    silence = np.zeros(int(0.2 * SAMPLE_RATE))  # 200ms silence
    full_audio = np.concatenate(
        [np.concatenate([a, silence]) for a in audio_arrays]
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, SAMPLE_RATE, (full_audio * 32767).astype(np.int16))

    console.print(f"[green]Saved audio to {output_path}[/green]")
    return output_path


def _split_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text into sentences for better Bark processing."""
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Merge very short sentences, split very long ones
    result = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                result.append(current)
            current = sentence

    if current:
        result.append(current)

    return result if result else [text]

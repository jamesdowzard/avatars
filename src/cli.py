"""CLI for avatar generation pipeline."""
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .config import (
    get_executive,
    get_executive_photo,
    list_executives,
    DATA_DIR,
    EXECUTIVES_DIR,
)

console = Console()


@click.group()
def cli():
    """Avatar generation pipeline - create animated talking avatars."""
    pass


@cli.command()
def list():
    """List available executives."""
    execs = list_executives()
    if not execs:
        console.print("[yellow]No executives configured yet.[/yellow]")
        console.print(f"Add executives to: {EXECUTIVES_DIR}/")
        return

    table = Table(title="Available Executives")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Title")

    for exec_id in execs:
        meta = get_executive(exec_id)
        table.add_row(exec_id, meta["name"], meta.get("title", ""))

    console.print(table)


@cli.command()
@click.argument("exec_id")
@click.argument("text")
@click.option("--style", default="illustration", help="Style preset (pixar, illustration, cartoon)")
@click.option("--voice", default="male_british", help="Voice preset")
@click.option("--output", "-o", type=Path, help="Output video path")
@click.option("--skip-style", is_flag=True, help="Use original photo (no style transfer)")
def generate(exec_id: str, text: str, style: str, voice: str, output: Path, skip_style: bool):
    """
    Generate a talking avatar video.

    EXEC_ID: Executive identifier (e.g., 'steve-butcher')
    TEXT: Text for the avatar to speak
    """
    from .tts.replicate_tts import generate_speech_replicate
    from .style.replicate_style import stylize_with_replicate
    from .animation.replicate_backend import animate_with_replicate

    # Get executive photo
    console.print(f"[bold]Generating avatar video for {exec_id}[/bold]\n")
    photo_path = get_executive_photo(exec_id)
    console.print(f"Source photo: {photo_path}")

    # Output paths
    output = output or DATA_DIR / "output" / f"{exec_id}_video.mp4"
    audio_path = DATA_DIR / "audio" / f"{exec_id}_speech.wav"
    styled_path = DATA_DIR / "stylized" / f"{exec_id}_{style}.png"

    # Step 1: Style transfer (optional)
    if skip_style:
        image_for_animation = photo_path
        console.print("[dim]Skipping style transfer (using original photo)[/dim]")
    else:
        console.print(f"\n[bold]Step 1/3: Style Transfer[/bold]")
        image_for_animation = stylize_with_replicate(photo_path, styled_path, style=style)

    # Step 2: Generate speech
    console.print(f"\n[bold]Step 2/3: Text-to-Speech[/bold]")
    console.print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    generate_speech_replicate(text, audio_path, voice=voice)

    # Step 3: Animate
    console.print(f"\n[bold]Step 3/3: Animation[/bold]")
    animate_with_replicate(image_for_animation, audio_path, output)

    console.print(f"\n[bold green]Done! Video saved to: {output}[/bold green]")


@cli.command()
@click.argument("exec_id")
@click.option("--style", default="illustration", help="Style preset")
@click.option("--output", "-o", type=Path, help="Output image path")
def stylize(exec_id: str, style: str, output: Path):
    """Generate a stylized avatar image (no animation)."""
    from .style.replicate_style import stylize_with_replicate

    photo_path = get_executive_photo(exec_id)
    output = output or DATA_DIR / "stylized" / f"{exec_id}_{style}.png"

    stylize_with_replicate(photo_path, output, style=style)


@cli.command()
@click.argument("text")
@click.option("--voice", default="male_british", help="Voice preset")
@click.option("--output", "-o", type=Path, required=True, help="Output audio path")
def speak(text: str, voice: str, output: Path):
    """Generate speech audio from text."""
    from .tts.replicate_tts import generate_speech_replicate

    generate_speech_replicate(text, output, voice=voice)


@cli.command()
@click.argument("image", type=Path)
@click.argument("audio", type=Path)
@click.option("--output", "-o", type=Path, required=True, help="Output video path")
@click.option("--model", default="liveportrait", help="Animation model (sadtalker, liveportrait)")
def animate(image: Path, audio: Path, output: Path, model: str):
    """Animate an image with audio to create talking video."""
    from .animation.replicate_backend import animate_with_replicate

    animate_with_replicate(image, audio, output, model=model)


if __name__ == "__main__":
    cli()

# Avatars

AI avatar generation pipeline - create animated talking avatars from photos.

## Pipeline

```
Photo → Style Transfer → TTS Audio → Animation → Talking Video
```

## Setup

```bash
# Install
uv sync

# Set Replicate API token
export REPLICATE_API_TOKEN=your_token_here
```

Get a token at: https://replicate.com/account/api-tokens

## Usage

### Full Pipeline

Generate a complete talking avatar video:

```bash
# Using an executive's photo
avatars generate steve-butcher "Welcome to the Q3 update. We've made significant progress on the metro project."

# With options
avatars generate steve-butcher "Hello everyone" --style pixar --voice male_british -o output.mp4

# Skip style transfer (use original photo)
avatars generate steve-butcher "Hello" --skip-style
```

### Individual Steps

```bash
# List available executives
avatars list

# Just stylize a photo
avatars stylize steve-butcher --style illustration

# Just generate speech
avatars speak "Hello world" --voice male_british -o speech.wav

# Just animate (image + audio → video)
avatars animate photo.jpg audio.wav -o video.mp4
```

## Style Presets

- `illustration` - Clean corporate illustration style
- `pixar` - 3D Pixar-style character
- `cartoon` - Flat cartoon/vector style

## Voice Presets

- `male_british` - British male voice
- `male_american` - American male voice
- `female_british` - British female voice
- `female_american` - American female voice

## Adding Executives

Create a folder in `executives/` with:

```
executives/
  john-smith/
    metadata.json
    photo.jpg
```

metadata.json:
```json
{
  "id": "john-smith",
  "name": "John Smith",
  "title": "CEO",
  "company": "Acme Corp",
  "photos": {
    "primary": "photo.jpg"
  }
}
```

## Models Used

- **Style Transfer**: Stable Diffusion XL (img2img)
- **TTS**: Bark (Suno AI)
- **Animation**: LivePortrait / SadTalker

All models run via Replicate API by default. Local execution support planned.

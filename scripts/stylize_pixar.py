"""Generate Pixar-style avatar using IP-Adapter with face focus."""
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import argparse
import os

def generate_pixar_avatar(
    image_path: str,
    output_path: str,
    strength: float = 0.55,
):
    """Generate a Pixar-style avatar from a photo using img2img."""
    print("Loading SDXL img2img pipeline...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    # Load and resize input image
    print(f"Loading input image: {image_path}")
    init_image = load_image(image_path).convert("RGB")
    init_image = init_image.resize((1024, 1024))

    prompt = """pixar style 3D animated character, disney pixar movie character portrait,
    friendly middle-aged businessman executive, wearing dark suit and white shirt,
    warm smile, glasses, short grey hair, professional corporate look,
    simple gradient background, high quality 3D render, detailed face,
    soft studio lighting, octane render"""

    negative_prompt = """realistic photo, photograph, blurry, low quality,
    distorted face, ugly, deformed, bad anatomy, extra limbs,
    text, watermark, signature"""

    print(f"Generating Pixar-style avatar (strength={strength})...")
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=40,
        negative_prompt=negative_prompt,
    ).images[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input photo path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--strength", "-s", type=float, default=0.55,
                        help="Transform strength (0.0-1.0, higher=more stylized)")
    args = parser.parse_args()

    generate_pixar_avatar(args.input, args.output, strength=args.strength)

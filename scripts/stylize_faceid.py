"""Generate Pixar-style avatar using IP-Adapter FaceID for strong identity preservation."""
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import argparse
import os

def get_face_embedding(image_path, app):
    """Extract face embedding from image."""
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in image")
    return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

def generate_faceid_avatar(
    image_path: str,
    output_path: str,
    ip_scale: float = 0.7,
):
    """Generate a Pixar-style avatar using IP-Adapter FaceID."""

    print("Loading face analysis model...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    print(f"Extracting face embedding from {image_path}...")
    face_emb = get_face_embedding(image_path, app)

    print("Downloading IP-Adapter FaceID...")
    ip_ckpt = hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sdxl.bin",
        local_dir="./checkpoints"
    )

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load IP-Adapter FaceID
    print("Loading IP-Adapter FaceID weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sdxl.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(ip_scale)

    # Prepare face embedding
    face_emb = face_emb.to("cuda", dtype=torch.float16)

    prompt = """pixar disney 3D animated movie character, portrait of friendly middle-aged
    executive businessman, dark navy suit, white dress shirt, genuine warm smile,
    rimless rectangular glasses, short grey-brown hair, professional look,
    soft gradient background, high quality 3D render, detailed expressive face,
    pixar movie quality, dreamworks animation style, soft studio lighting"""

    negative_prompt = """realistic photo, photograph, blurry, low quality, distorted face,
    ugly, deformed, bad anatomy, poorly drawn, mutation, extra limbs,
    text, watermark, 2D, flat, cartoon sketch"""

    print(f"Generating with IP-Adapter scale={ip_scale}...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image_embeds=[face_emb],
        num_inference_steps=30,
        guidance_scale=6.0,
        width=1024,
        height=1024,
    ).images[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)
    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input photo path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--scale", "-s", type=float, default=0.7,
                        help="IP-Adapter scale (0.0-1.0, higher=stronger likeness)")
    args = parser.parse_args()

    generate_faceid_avatar(args.input, args.output, ip_scale=args.scale)

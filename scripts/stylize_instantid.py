"""Generate Pixar-style avatar using InstantID for strong likeness preservation."""
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, ControlNetModel, DDIMScheduler
from diffusers.pipelines import StableDiffusionXLControlNetPipeline
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import argparse
import os

def download_instantid_models():
    """Download InstantID model files."""
    # Download ControlNet
    controlnet_path = hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="./checkpoints",
    )
    controlnet_config = hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="./checkpoints",
    )

    # Download IP-Adapter
    ip_adapter_path = hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="./checkpoints",
    )

    return os.path.dirname(controlnet_path), ip_adapter_path

def get_face_info(image_path, app):
    """Extract face embedding and keypoints from image."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in image")

    face = faces[0]
    # Get face embedding
    face_emb = face.normed_embedding
    # Get face keypoints for ControlNet
    face_kps = face.kps

    return face_emb, face_kps, img_rgb

def draw_kps(image_shape, kps):
    """Draw keypoints as a control image."""
    h, w = image_shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw keypoints
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(out, (x, y), 3, (255, 255, 255), -1)

    return Image.fromarray(out)

def generate_instantid_avatar(
    image_path: str,
    output_path: str,
    prompt: str = None,
):
    """Generate a Pixar-style avatar preserving identity with InstantID."""

    print("Loading face analysis model...")
    app = FaceAnalysis(
        name="antelopev2",
        root="./",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Extracting face information...")
    face_emb, face_kps, img_rgb = get_face_info(image_path, app)

    print("Downloading InstantID models...")
    controlnet_dir, ip_adapter_path = download_instantid_models()

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_dir,
        torch_dtype=torch.float16,
    )

    print("Loading SDXL pipeline with InstantID...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load IP-Adapter for face identity
    pipe.load_ip_adapter_face_id(ip_adapter_path)

    # Prepare face embedding
    face_emb_tensor = torch.tensor(face_emb).unsqueeze(0).to("cuda", dtype=torch.float16)

    # Create control image from keypoints
    control_image = draw_kps(img_rgb.shape, face_kps)
    control_image = control_image.resize((1024, 1024))

    if prompt is None:
        prompt = """pixar disney 3D animated character portrait, friendly executive businessman,
        wearing dark navy suit and white dress shirt, warm genuine smile, rimless glasses,
        short grey hair, professional corporate look, simple soft gradient background,
        high quality 3D render, detailed expressive face, soft studio lighting,
        pixar movie style, dreamworks animation quality"""

    negative_prompt = """realistic photo, photograph, blurry, low quality, distorted,
    ugly, deformed, bad anatomy, disfigured, poorly drawn face, mutation,
    extra limbs, text, watermark, signature, 2D, flat"""

    print("Generating Pixar-style avatar with identity preservation...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        ip_adapter_face_id_embeds=face_emb_tensor,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5.0,
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
    args = parser.parse_args()

    generate_instantid_avatar(args.input, args.output)

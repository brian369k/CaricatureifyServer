# Caricatureify - Windows GPU Server (FastAPI + Diffusers, SDXL img2img with optional LoRA)
# Runs locally on http://0.0.0.0:5000
# Place any LoRA .safetensors in ./loras/<StyleName>/   (e.g., ./loras/PencilCaricature/your_lora.safetensors)

import io, os, random, time
from typing import Optional, Dict, Any
from PIL import Image, ImageOps
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --------- MODEL SETUP ---------
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Configurable paths
BASE_MODEL = os.environ.get("CARICATUREIFY_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_MODEL  = os.environ.get("CARICATUREIFY_VAE",   "madebyollin/sdxl-vae-fp16-fix")
LORAS_DIR  = os.path.abspath(os.environ.get("CARICATUREIFY_LORAS", "./loras"))

# Style presets (can be edited while server runs)
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Pencil Caricature": {
        "prompt": "award-winning pencil caricature portrait, exaggerated facial features, deep graphite shading, crosshatching, 3D volume, fine details, studio lighting, highly detailed, realistic, masterpiece",
        "negative": "blurry, lowres, bad anatomy, extra limbs, watermark, text, logo",
        "lora": "PencilCaricature",
        "strength": 0.45,
        "guidance": 7.5
    },
    "Ink Comic": {
        "prompt": "inked comic-book caricature, bold linework, clean inks, cel shading, high contrast, dynamic, expressive, professional illustration",
        "negative": "low contrast, smudged, blurry, watermark, text, logo",
        "lora": "InkComic",
        "strength": 0.5,
        "guidance": 8.0
    },
    "Disney Toon": {
        "prompt": "pixar disney style toddler caricature, cute face, big eyes, soft shading, smooth skin, vibrant colors, studio character render, highly polished",
        "negative": "grain, noise, lowres, watermark, text, extra fingers",
        "lora": "DisneyToon",
        "strength": 0.55,
        "guidance": 7.0
    },
    "Hyperreal Baby Version": {
        "prompt": "hyperrealistic baby version caricature, realistic skin, soft cheeks, perfect lighting, ultra-detailed face, studio portrait, shallow depth of field",
        "negative": "deformed, distorted, watermark, text, logo",
        "lora": "BabyVersion",
        "strength": 0.4,
        "guidance": 6.5
    },
    "Halloween Witch Scene": {
        "prompt": "Halloween witch theme caricature, moonlit sky, pumpkins, spooky background, cinematic lighting, detailed graphite and charcoal look, dramatic",
        "negative": "low detail, noisy, watermark, text",
        "lora": "WitchTheme",
        "strength": 0.6,
        "guidance": 8.5
    }
}

def list_lora_files(style_name: str):
    style_path = os.path.join(LORAS_DIR, style_name)
    if not os.path.isdir(style_path):
        return []
    return [os.path.join(style_path, f) for f in os.listdir(style_path) if f.lower().endswith(".safetensors")]

def load_pipeline():
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=DTYPE)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        add_watermarker=False,
        vae=vae,
        use_safetensors=True
    )
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe.to(DEVICE)
    pipe.safety_checker = None  # local / offline
    return pipe

PIPE = load_pipeline()

def apply_lora(style_name: str, scale: float = 1.0):
    # Load the first LoRA file found for this style (if present)
    lorafiles = list_lora_files(style_name)
    if not lorafiles:
        return None
    lora_path = lorafiles[0]
    try:
        PIPE.load_lora_weights(lora_path)  # diffusers >= 0.22 supports this
        PIPE.fuse_lora(lora_scale=scale)
        return lora_path
    except Exception as e:
        print(f"[WARN] Failed to load LoRA {lora_path}: {e}")
        return None

# ---------- FASTAPI ----------
app = FastAPI(title="Caricatureify Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/styles")
def styles():
    # Also indicate if a LoRA file is present for each style
    return {name: {"has_lora": len(list_lora_files(v.get("lora", name))) > 0} for name, v in STYLE_PRESETS.items()}

def preprocess(image: Image.Image, max_side: int = 1024) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return image

@app.post("/caricature")
def caricature(
    image: UploadFile = File(...),
    style: str = Form("Pencil Caricature"),
    prompt: Optional[str] = Form(None),
    negative: Optional[str] = Form(None),
    strength: float = Form(0.5),
    guidance: float = Form(7.5),
    seed: Optional[int] = Form(None),
    steps: int = Form(30),
    lora_scale: float = Form(0.9),
):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(image.file.read()))
        img = preprocess(img)

        # Compose prompts
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Pencil Caricature"])
        full_prompt = (preset["prompt"] + (", " + prompt if prompt else "")).strip()
        full_negative = (preset.get("negative","") + (", " + negative if negative else "")).strip()

        # Apply LoRA if present
        lora_used = apply_lora(preset.get("lora", style), scale=float(lora_scale))

        generator = torch.Generator(device=DEVICE)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        generator.manual_seed(int(seed))

        with torch.inference_mode():
            out = PIPE(
                prompt=full_prompt,
                negative_prompt=full_negative,
                image=img,
                strength=float(strength),
                guidance_scale=float(guidance),
                num_inference_steps=int(steps),
                generator=generator,
            ).images[0]

        # JPEG encode
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        headers = {
            "x-seed": str(seed),
            "x-style": style,
            "x-lora": lora_used or "",
            "x-latency-ms": str(int((time.time()-t0)*1000))
        }
        return StreamingResponse(buf, media_type="image/jpeg", headers=headers)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
Caricatureify - Windows GPU Server (RTX 3070 Ready)
===================================================

1) Install Python 3.11 (64-bit) if you don't have it.
2) Double-click run_server.bat (first run will download models and dependencies).
3) The server runs at: http://<YOUR-PC-IP>:5000
   - Test in browser: http://<YOUR-PC-IP>:5000/styles

LoRA styles
-----------
Drop LoRA .safetensors into subfolders inside /loras:

  /loras/PencilCaricature/your_pencil_lora.safetensors
  /loras/InkComic/your_ink_lora.safetensors
  /loras/DisneyToon/your_disney_lora.safetensors
  /loras/BabyVersion/your_baby_lora.safetensors
  /loras/WitchTheme/your_witch_lora.safetensors

(If a folder is empty, the style still works using the base SDXL prompts.)

API
---
POST /caricature (multipart/form-data)
  - image: file
  - style: one of /styles
  - prompt (optional): extra prompt
  - negative (optional)
  - strength (0.1..0.9): higher = more stylized
  - guidance (5..12): higher = follows prompt more
  - steps (20..40): more steps = more detail
  - seed (int, optional)
  - lora_scale (0.1..1.2)

Returns: JPEG image body with headers:
  x-seed, x-style, x-lora, x-latency-ms

Troubleshooting
---------------
- Speed: first run is slow due to model downloads and GPU warm-up. Later runs are fast.
- If CUDA OOM: reduce image size or strength/steps.
- To force local-only (no downloads), pre-cache models via 'huggingface-cli download' and set env vars in run_server.bat to folders.
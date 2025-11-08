@echo off
title Caricatureify Server
call "%~dp0venv\Scripts\activate.bat" 2>NUL || (
  echo Creating virtual environment...
  py -3 -m venv "%~dp0venv"
  call "%~dp0venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r "%~dp0requirements.txt"
)
set CARICATUREIFY_LORAS=%~dp0loras
set CARICATUREIFY_MODEL=stabilityai/stable-diffusion-xl-base-1.0
set CARICATUREIFY_VAE=madebyollin/sdxl-vae-fp16-fix
python "%~dp0caricature_server.py"
pause
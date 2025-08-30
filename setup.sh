#!/usr/bin/env bash
set -e

PY311=$(command -v python3.11 || true)
if [ -n "$PY311" ]; then
  "$PY311" -m venv .venv
else
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# Core torch stack pinned for macOS ARM
pip install --no-cache-dir torch==2.1.1 torchaudio==2.1.1

# Install audiocraft without heavy extras (avoid xformers)
pip install --no-cache-dir --no-deps audiocraft==1.3.0

# Essential runtime deps for MusicGen path in audiocraft
pip install --no-cache-dir encodec==0.1.1 av==11.0.0 pyloudnorm sentencepiece einops hydra-core hydra_colorlog julius num2words huggingface_hub numpy spacy

# Project deps
pip install --no-cache-dir transformers matplotlib pandas scipy scikit-learn tqdm

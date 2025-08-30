from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import json
import torch
import argparse

parser = argparse.ArgumentParser(description='Generate music for a specific artist')
parser.add_argument('--artist_name', required=True, help='Name of the artist (e.g., billie)')
args = parser.parse_args()

artist = args.artist_name

OUT = Path("gen")
OUT.mkdir(exist_ok=True, parents=True)
cfg = json.loads(Path("tokens.json").read_text())

# Get artist-specific config
if artist in cfg:
    artist_cfg = cfg[artist]
else:
    raise ValueError(f"Artist '{artist}' not found in tokens.json. Available: {list(cfg.keys())}")

baseline = artist_cfg["baseline"]
sets = artist_cfg["sets"]

model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
model.set_generation_params(duration=15)

def seeds():
    for s in range(10):
        yield s

# baseline once
bdir = OUT / "baseline" / artist
bdir.mkdir(parents=True, exist_ok=True)
for s in seeds():
    torch.manual_seed(s)
    wav = model.generate([baseline])[0, 0].cpu()
    audio_write(bdir / f"seed_{s:03d}", wav, 32000, strategy="loudness")

# styled per set
for i, toks in enumerate(sets, start=1):
    sdir = OUT / f"set{i}" / artist
    sdir.mkdir(parents=True, exist_ok=True)
    styled = baseline + ", " + ", ".join(toks)
    for s in seeds():
        torch.manual_seed(s)
        wav = model.generate([styled])[0, 0].cpu()
        audio_write(sdir / f"seed_{s:03d}", wav, 32000, strategy="loudness")

print(f"✓ Generated baseline and 5 styled sets for {artist}")

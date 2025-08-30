from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import json
import torch
import argparse

parser = argparse.ArgumentParser(description='Generate artist name baseline clips')
parser.add_argument('--artist_name', required=True, help='Name of the artist (e.g., billie)')
args = parser.parse_args()

artist = args.artist_name

OUT = Path("gen")
OUT.mkdir(exist_ok=True, parents=True)
cfg = json.loads(Path("tokens.json").read_text())

if artist in cfg:
    artist_cfg = cfg[artist]
else:
    raise ValueError(f"Artist '{artist}' not found in tokens.json. Available: {list(cfg.keys())}")

baseline = artist_cfg["baseline"]
artist_name = artist_cfg["artist_name"]

# Create artist name baseline prompt
artist_baseline = f"{baseline} [{artist_name}]"

model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
model.set_generation_params(duration=15)

def seeds():
    for s in range(10):
        yield s

# Generate artist_name_baseline clips
artist_name_dir = OUT / "artist_name_baseline" / artist
artist_name_dir.mkdir(exist_ok=True, parents=True)

print(f"Generating artist_name_baseline for {artist}")
print(f"Prompt: {artist_baseline}")

for s in seeds():
    torch.manual_seed(s)
    wav = model.generate([artist_baseline])[0, 0].cpu()
    audio_write(artist_name_dir / f"seed_{s:03d}", wav, 32000, strategy="loudness")
    print(f"  Generated seed_{s:03d}")

print(f"Generated 10 artist_name_baseline clips for {artist}")


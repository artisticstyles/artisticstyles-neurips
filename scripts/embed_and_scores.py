import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import torchaudio, torch, pandas as pd, tqdm
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate embeddings and scores for a specific artist')
parser.add_argument('--artist_name', required=True, help='Name of the artist (e.g., billie)')
args = parser.parse_args()

artist = args.artist_name

# Load VGGish model
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

def embed(path: Path):
    wav, sr = torchaudio.load(str(path))
    wav = wav.numpy().flatten()
    
    # Resample to 16kHz for VGGish
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    
    # VGGish expects float32 input
    wav = wav.astype(np.float32)
    
    # Get embeddings
    embeddings = vggish_model(wav)
    
    embeddings = torch.from_numpy(embeddings.numpy())
    return embeddings.mean(0)

# reference - look for artist-specific folder first, fallback to generic
ref_dir = Path("reference") / artist
if not ref_dir.exists():
    ref_dir = Path("reference")
    
ref_embs = torch.stack([embed(p) for p in sorted(ref_dir.glob("*.wav"))])
ref_centroid = ref_embs.mean(0, keepdim=True)
ref_norm = torch.nn.functional.normalize(ref_centroid, dim=-1)

# generated
rows = []
embeddings = {"ref": ref_embs}
gen_root = Path("gen")

def cos_to_ref(z):
    z = z.unsqueeze(0)
    z = torch.nn.functional.normalize(z, dim=-1)
    ref_norm_local = torch.nn.functional.normalize(ref_centroid, dim=-1)
    return torch.nn.functional.cosine_similarity(z, ref_norm_local).item()

# baseline
bdir = gen_root / "baseline" / artist
b_embs = []
for f in sorted(bdir.glob("*.wav")):
    z = embed(f)
    b_embs.append(z)
    rows.append({"split": "baseline", "file": f.name, "vggish_cos": cos_to_ref(z)})
embeddings["baseline"] = torch.stack(b_embs)

# sets
embeddings["sets"] = {}
for i in range(1, 6):  # set1 through set5
    sdir = gen_root / f"set{i}" / artist
    if sdir.exists():
        s_embs = []
        for f in sorted(sdir.glob("*.wav")):
            z = embed(f)
            s_embs.append(z)
            rows.append({"split": f"set{i}", "file": f.name, "vggish_cos": cos_to_ref(z)})
        embeddings["sets"][f"set{i}"] = torch.stack(s_embs)

# Create results directory
import os
results_dir = Path("results") / artist
results_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(results_dir / "vggish_scores.csv", index=False)
torch.save({"ref": embeddings["ref"],
            "baseline": embeddings["baseline"],
            "sets": embeddings["sets"]}, results_dir / "embeddings.pt")
print(f"✓ Saved results/{artist}/vggish_scores.csv and results/{artist}/embeddings.pt")

import tensorflow as tf
import tensorflow_hub as hub
from msclap import CLAP
from pathlib import Path
import torchaudio, torch, pandas as pd, tqdm
import numpy as np
import argparse
import librosa

parser = argparse.ArgumentParser(description='Extract VGGish and CLAP embeddings for all clips')
parser.add_argument('--artist_name', required=True, help='Name of the artist (e.g., billie)')
args = parser.parse_args()

artist = args.artist_name

# Load models
print("Loading VGGish model...")
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

print("Loading Microsoft CLAP model...")
clap_model = CLAP(version='2023', use_cuda=False)

def embed_vggish(path: Path):
    wav, sr = torchaudio.load(str(path))
    wav = wav.numpy().flatten()
    
    # Resample to 16kHz for VGGish
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    
    # VGGish expects float32 input
    wav = wav.astype(np.float32)
    
    # Get embeddings
    embeddings = vggish_model(wav)
    embeddings = torch.from_numpy(embeddings.numpy())
    return embeddings.mean(0)

def embed_clap(path: Path):
    # CLAP expects file paths, will handle loading and resampling internally
    audio_embed = clap_model.get_audio_embeddings([str(path)], resample=True)
    # CLAP returns tensor, convert to numpy then back to tensor for consistency
    if isinstance(audio_embed, torch.Tensor):
        return audio_embed[0]
    else:
        return torch.from_numpy(audio_embed[0])

# Create output directories
for embed_type in ['vggish', 'clap']:
    for condition in ['references', 'baseline', 'set1', 'set2', 'set3', 'set4', 'set5', 'artist_name_baseline']:
        (Path("embeddings") / embed_type / artist / condition).mkdir(exist_ok=True, parents=True)

# Process references
ref_dir = Path("reference") / artist
if not ref_dir.exists(): 
    ref_dir = Path("reference")

print(f"Processing references for {artist}...")
for i, ref_path in enumerate(sorted(ref_dir.glob("*.wav"))):
    print(f"  {ref_path.name}")
    
    # VGGish
    vgg_emb = embed_vggish(ref_path)
    np.save(Path("embeddings") / "vggish" / artist / "references" / f"ref_{i:03d}.npy", vgg_emb.numpy())
    
    # CLAP
    clap_emb = embed_clap(ref_path)
    np.save(Path("embeddings") / "clap" / artist / "references" / f"ref_{i:03d}.npy", clap_emb.numpy())

# Process all generated conditions
conditions = ['baseline', 'set1', 'set2', 'set3', 'set4', 'set5', 'artist_name_baseline']

for condition in conditions:
    gen_dir = Path("gen") / condition / artist
    if not gen_dir.exists():
        print(f"Skipping {condition} - directory not found")
        continue
        
    print(f"Processing {condition} for {artist}...")
    for wav_path in sorted(gen_dir.glob("seed_*.wav")):
        seed_name = wav_path.stem
        print(f"  {seed_name}")
        
        # VGGish
        vgg_emb = embed_vggish(wav_path)
        np.save(Path("embeddings") / "vggish" / artist / condition / f"{seed_name}.npy", vgg_emb.numpy())
        
        # CLAP
        clap_emb = embed_clap(wav_path)
        np.save(Path("embeddings") / "clap" / artist / condition / f"{seed_name}.npy", clap_emb.numpy())

print(f"Dual embedding extraction complete for {artist}")

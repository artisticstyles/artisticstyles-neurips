import torch, numpy as np, pandas as pd
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate distribution metrics for a specific artist')
parser.add_argument('--artist_name', required=True, help='Name of the artist (e.g., billie)')
args = parser.parse_args()

artist = args.artist_name

from pathlib import Path
data = torch.load(Path("results") / artist / "embeddings.pt")
E_ref = data["ref"].numpy()
E_base = data["baseline"].numpy()
E_sets = {k: v.numpy() for k, v in data["sets"].items()}

def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

def frechet(A,B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    mu1, mu2 = A.mean(0), B.mean(0)
    C1, C2 = np.cov(A.T), np.cov(B.T)
    diff = mu1 - mu2
    covmean = sqrtm(C1.dot(C2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff@diff + np.trace(C1 + C2 - 2*covmean))

def pr_metrics(real, gen, quantile=95):
    nn_rr = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(real)
    d_rr, _ = nn_rr.kneighbors(real)
    radius = np.percentile(d_rr[:,1], quantile)
    
    nn_rg = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real)
    d_g2r, _ = nn_rg.kneighbors(gen)
    precision = float(np.mean(d_g2r[:,0] <= radius))
    
    nn_gr = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(gen)
    d_r2g, _ = nn_gr.kneighbors(real)
    recall = float(np.mean(d_r2g[:,0] <= radius))
    return precision, recall

def diversity(X):
    return float(pdist(X, metric="cosine").mean())

R = l2norm(E_ref)
B = l2norm(E_base)
S = {k: l2norm(v) for k,v in E_sets.items()}

rows = []
p,r = pr_metrics(R,B)
rows.append({"set":"baseline",
             "FCD": frechet(B,R),
             "Precision": p,
             "Recall": r,
             "Diversity": diversity(B)})

for k, V in S.items():
    p,r = pr_metrics(R,V)
    rows.append({"set":k,
                 "FCD": frechet(V,R),
                 "Precision": p,
                 "Recall": r,
                 "Diversity": diversity(V)})

df = pd.DataFrame(rows).sort_values("set")

# Create results directory  
results_dir = Path("results") / artist
results_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(results_dir / "metrics.csv", index=False)

best = df[df["set"]!="baseline"].sort_values("FCD").iloc[0]["set"]
print(f"Best set by FCD for {artist}:", best)
print(f"✓ Saved results/{artist}/metrics.csv")

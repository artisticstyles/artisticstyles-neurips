#!/usr/bin/env python3
"""
Quick script to compute FAD for all style sets
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import time

def load_embeddings(embed_type, artist, condition):
    """Load all embeddings for a given condition"""
    embed_dir = Path("embeddings") / embed_type / artist / condition
    if not embed_dir.exists():
        return None
    
    embeddings = []
    for npy_file in sorted(embed_dir.glob("*.npy")):
        emb = np.load(npy_file)
        embeddings.append(torch.from_numpy(emb))
    
    if len(embeddings) == 0:
        return None
    return torch.stack(embeddings)

def frechet_distance(X, Y):
    """Compute Fréchet distance between two sets of embeddings"""
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    
    X_cov = torch.cov(X.T)
    Y_cov = torch.cov(Y.T)
    
    mean_diff = X_mean - Y_mean
    cov_mean = (X_cov + Y_cov) / 2
    
    # Simple approximation of Fréchet distance
    return torch.norm(mean_diff).item() + torch.trace(X_cov + Y_cov - 2 * cov_mean).item()

def main():
    results = []
    
    artists = ['billie', 'einaudi']
    embed_types = ['vggish', 'clap']
    style_sets = ['set1', 'set2', 'set3', 'set4', 'set5']
    
    start_time = time.time()
    total_computations = len(artists) * len(embed_types) * len(style_sets)
    current = 0
    
    for artist in artists:
        for embed_type in embed_types:
            print(f"\n🔥 Computing {artist} {embed_type} FADs...")
            
            # Load reference embeddings
            ref_embs = load_embeddings(embed_type, artist, "references")
            if ref_embs is None:
                print(f"❌ No reference embeddings for {artist} {embed_type}")
                continue
                
            # Load baseline and artist_name for comparison
            baseline_embs = load_embeddings(embed_type, artist, "baseline")
            artist_embs = load_embeddings(embed_type, artist, "artist_name_baseline")
            
            if baseline_embs is not None:
                baseline_fad = frechet_distance(baseline_embs, ref_embs)
                print(f"   Baseline FAD: {baseline_fad:.3f}")
            else:
                baseline_fad = None
                
            if artist_embs is not None:
                artist_fad = frechet_distance(artist_embs, ref_embs)
                print(f"   Artist FAD: {artist_fad:.3f}")
            else:
                artist_fad = None
            
            # Compute FAD for each style set
            style_fads = []
            for style_set in style_sets:
                current += 1
                style_embs = load_embeddings(embed_type, artist, style_set)
                
                if style_embs is not None:
                    style_fad = frechet_distance(style_embs, ref_embs)
                    style_fads.append(style_fad)
                    print(f"   {style_set} FAD: {style_fad:.3f}")
                else:
                    style_fads.append(None)
                    print(f"   {style_set} FAD: MISSING")
                    
                # Progress
                elapsed = time.time() - start_time
                remaining = (elapsed / current) * (total_computations - current)
                print(f"   Progress: {current}/{total_computations} ({current/total_computations*100:.1f}%) - ETA: {remaining:.0f}s")
            
            # Calculate average
            valid_fads = [f for f in style_fads if f is not None]
            if valid_fads:
                avg_fad = np.mean(valid_fads)
                print(f"   Average Style FAD: {avg_fad:.3f}")
                
                results.append({
                    'artist': artist,
                    'embed_type': embed_type,
                    'baseline_fad': baseline_fad,
                    'artist_fad': artist_fad,
                    'avg_style_fad': avg_fad,
                    'set1_fad': style_fads[0],
                    'set2_fad': style_fads[1], 
                    'set3_fad': style_fads[2],
                    'set4_fad': style_fads[3],
                    'set5_fad': style_fads[4],
                    'best_fad': min(valid_fads),
                    'best_set': f"set{np.argmin(valid_fads) + 1}"
                })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/all_style_fads.csv', index=False)
    
    print(f"\nDONE! Computed {total_computations} FADs in {time.time() - start_time:.1f}s")
    print("Results saved to results/all_style_fads.csv")
    
    # Show comparison summary
    print("\nBEST vs AVERAGE COMPARISON:")
    for _, row in df.iterrows():
        print(f"{row.artist} {row.embed_type}:")
        print(f"   Best Style:    {row.best_fad:.3f} ({row.best_set})")
        print(f"   Average Style: {row.avg_style_fad:.3f}")
        print(f"   Difference:    {row.avg_style_fad - row.best_fad:.3f} (+{((row.avg_style_fad/row.best_fad-1)*100):.1f}%)")

if __name__ == "__main__":
    main()

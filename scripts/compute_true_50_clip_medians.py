#!/usr/bin/env python3
"""
Compute true median min-distances using all 50 style clips
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

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

def compute_min_distances(gen_embeddings, ref_embeddings):
    """Compute min-distance for each generated embedding to reference set"""
    min_distances = []
    
    for gen_emb in gen_embeddings:
        # Compute cosine similarities to all references
        similarities = torch.nn.functional.cosine_similarity(
            gen_emb.unsqueeze(0), ref_embeddings, dim=1
        )
        # Convert to distances (1 - similarity) and take minimum
        distances = 1 - similarities
        min_dist = torch.min(distances).item()
        min_distances.append(min_dist)
    
    return np.array(min_distances)

def main():
    results = []
    
    artists = ['billie', 'einaudi']
    embed_types = ['vggish', 'clap']
    style_sets = ['set1', 'set2', 'set3', 'set4', 'set5']
    
    for artist in artists:
        for embed_type in embed_types:
            print(f"\n🔥 Computing true 50-clip medians for {artist} {embed_type}...")
            
            # Load reference embeddings
            ref_embs = load_embeddings(embed_type, artist, "references")
            if ref_embs is None:
                print(f"❌ No reference embeddings for {artist} {embed_type}")
                continue
            
            # Baseline and artist name (single conditions)
            baseline_embs = load_embeddings(embed_type, artist, "baseline")
            artist_embs = load_embeddings(embed_type, artist, "artist_name_baseline")
            
            baseline_median = None
            artist_median = None
            
            if baseline_embs is not None:
                baseline_dists = compute_min_distances(baseline_embs, ref_embs)
                baseline_median = np.median(baseline_dists)
                print(f"   Baseline: {baseline_median:.6f} (from {len(baseline_dists)} clips)")
            
            if artist_embs is not None:
                artist_dists = compute_min_distances(artist_embs, ref_embs)
                artist_median = np.median(artist_dists)
                print(f"   Artist: {artist_median:.6f} (from {len(artist_dists)} clips)")
            
            # Collect ALL style clip distances
            all_style_distances = []
            for style_set in style_sets:
                style_embs = load_embeddings(embed_type, artist, style_set)
                if style_embs is not None:
                    style_dists = compute_min_distances(style_embs, ref_embs)
                    all_style_distances.extend(style_dists)
                    print(f"   {style_set}: {np.median(style_dists):.6f} (from {len(style_dists)} clips)")
            
            # Compute TRUE median of all 50 style clips
            if len(all_style_distances) > 0:
                true_styled_median = np.median(all_style_distances)
                print(f"   ✨ TRUE Styled (all {len(all_style_distances)} clips): {true_styled_median:.6f}")
                
                results.append({
                    'artist': artist,
                    'embed_type': embed_type,
                    'baseline_median': baseline_median,
                    'artist_median': artist_median,
                    'styled_median_true': true_styled_median,
                    'num_style_clips': len(all_style_distances)
                })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/true_50_clip_medians.csv', index=False)
    
    print(f"\nDONE! True 50-clip medians computed")
    print("Results saved to results/true_50_clip_medians.csv")
    
    # Show comparison
    print("\nTRUE 50-CLIP MEDIANS:")
    for _, row in df.iterrows():
        print(f"{row.artist} {row.embed_type}:")
        print(f"   Baseline: {row.baseline_median:.6f}")
        print(f"   Artist:   {row.artist_median:.6f}")
        print(f"   Styled:   {row.styled_median_true:.6f} (from {row.num_style_clips} clips)")

if __name__ == "__main__":
    main()

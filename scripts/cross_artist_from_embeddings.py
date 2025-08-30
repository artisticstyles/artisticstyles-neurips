import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import wilcoxon

def load_embeddings_from_disk(embed_type, artist, condition):
    """Load pre-computed embeddings from disk"""
    embed_dir = Path("embeddings") / embed_type / artist / condition
    if not embed_dir.exists():
        print(f"    ⚠️  Directory not found: {embed_dir}")
        return None
    
    embeddings = []
    for npy_file in sorted(embed_dir.glob("*.npy")):
        emb = np.load(npy_file)
        embeddings.append(torch.from_numpy(emb))
    
    if not embeddings:
        print(f"    ⚠️  No embedding files found in: {embed_dir}")
        return None
    
    return torch.stack(embeddings)

def cos_similarity_batch(embs, ref_centroid):
    """Compute cosine similarity between embeddings and reference centroid"""
    embs_norm = torch.nn.functional.normalize(embs, dim=-1)
    ref_norm = torch.nn.functional.normalize(ref_centroid, dim=-1)
    return torch.nn.functional.cosine_similarity(embs_norm, ref_norm, dim=-1)

def main():
    print("Starting Cross-Artist Experiment from Pre-computed Embeddings...")
    print("This will compute cross-artist validation for both VGGish and CLAP embeddings")
    
    # Create output directories
    output_dir = Path("cross-artist-outputs")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    artists = ["billie", "einaudi"]
    embed_types = ["vggish", "clap"]
    
    all_results = []
    cross_artist_improvements = {}
    
    print("\nComputing cross-artist similarities for both embedding types...")
    
    for embed_type in embed_types:
        print(f"\nProcessing {embed_type.upper()} embeddings...")
        cross_artist_improvements[embed_type] = {}
        
        for gen_artist in artists:
            for ref_artist in artists:
                print(f"\n  {gen_artist.capitalize()} generation → {ref_artist.capitalize()} reference ({embed_type})")
                
                # Load reference embeddings
                ref_embs = load_embeddings_from_disk(embed_type, ref_artist, "references")
                if ref_embs is None:
                    continue
                
                ref_centroid = ref_embs.mean(0, keepdim=True)
                
                # Load generated embeddings - baseline
                baseline_embs = load_embeddings_from_disk(embed_type, gen_artist, "baseline")
                if baseline_embs is None:
                    continue
                
                # Load artist_name_baseline
                artist_name_embs = load_embeddings_from_disk(embed_type, gen_artist, "artist_name_baseline")
                
                # Load all styled sets and concatenate
                styled_embs_list = []
                for set_num in range(1, 6):  # sets 1-5
                    set_embs = load_embeddings_from_disk(embed_type, gen_artist, f"set{set_num}")
                    if set_embs is not None:
                        styled_embs_list.append(set_embs)
                
                if not styled_embs_list:
                    print(f"    ⚠️  No styled embeddings found for {gen_artist} {embed_type}")
                    continue
                
                styled_embs = torch.cat(styled_embs_list, dim=0)
                
                # Compute similarities
                baseline_sims = cos_similarity_batch(baseline_embs, ref_centroid).numpy()
                styled_sims = cos_similarity_batch(styled_embs, ref_centroid).numpy()
                
                # Store results for baseline
                for sim in baseline_sims:
                    all_results.append({
                        "embed_type": embed_type,
                        "generation_artist": gen_artist,
                        "reference_artist": ref_artist,
                        "condition": "baseline",
                        "is_same_artist": gen_artist == ref_artist,
                        "similarity": float(sim)
                    })
                
                # Store results for styled
                for sim in styled_sims:
                    all_results.append({
                        "embed_type": embed_type,
                        "generation_artist": gen_artist,
                        "reference_artist": ref_artist,
                        "condition": "styled",
                        "is_same_artist": gen_artist == ref_artist,
                        "similarity": float(sim)
                    })
                
                # Store results for artist_name if available
                if artist_name_embs is not None:
                    artist_name_sims = cos_similarity_batch(artist_name_embs, ref_centroid).numpy()
                    for sim in artist_name_sims:
                        all_results.append({
                            "embed_type": embed_type,
                            "generation_artist": gen_artist,
                            "reference_artist": ref_artist,
                            "condition": "artist_name",
                            "is_same_artist": gen_artist == ref_artist,
                            "similarity": float(sim)
                        })
                    print(f"    Artist:   {np.mean(artist_name_sims):.3f} ± {np.std(artist_name_sims):.3f}")
                
                # Compute improvement for this pair
                baseline_mean = np.mean(baseline_sims)
                styled_mean = np.mean(styled_sims)
                improvement = styled_mean - baseline_mean
                
                # Store for heatmap
                key = f"{gen_artist}_to_{ref_artist}"
                cross_artist_improvements[embed_type][key] = float(improvement)
                
                print(f"    Baseline: {baseline_mean:.3f} ± {np.std(baseline_sims):.3f}")
                print(f"    Styled:   {styled_mean:.3f} ± {np.std(styled_sims):.3f}")
                print(f"    Improvement: {improvement:+.3f}")
    
    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "dual_cross_artist_similarities.csv", index=False)
    print(f"\nSaved detailed results to {results_dir / 'dual_cross_artist_similarities.csv'}")
    
    # Create summary statistics
    print("\nComputing summary statistics...")
    summary_stats = []
    
    for embed_type in embed_types:
        embed_df = df[df.embed_type == embed_type]
        
        for gen_artist in artists:
            for ref_artist in artists:
                subset = embed_df[(embed_df.generation_artist == gen_artist) & 
                                (embed_df.reference_artist == ref_artist)]
                if len(subset) == 0:
                    continue
                
                baseline_subset = subset[subset.condition == "baseline"]
                styled_subset = subset[subset.condition == "styled"]
                
                if len(baseline_subset) == 0 or len(styled_subset) == 0:
                    continue
                
                # Compute improvement
                baseline_mean = baseline_subset.similarity.mean()
                styled_mean = styled_subset.similarity.mean()
                improvement = styled_mean - baseline_mean
                
                # Wilcoxon test
                try:
                    stat, p_val = wilcoxon(baseline_subset.similarity, 
                                         styled_subset.similarity, 
                                         alternative='two-sided')
                except:
                    p_val = np.nan
                
                summary_stats.append({
                    "Embedding": embed_type.upper(),
                    "Generation": gen_artist.capitalize(),
                    "Reference": ref_artist.capitalize(),
                    "Same Artist": "✓" if gen_artist == ref_artist else "✗",
                    "Baseline (mean ± SD)": f"{baseline_mean:.3f} ± {baseline_subset.similarity.std():.3f}",
                    "Styled (mean ± SD)": f"{styled_mean:.3f} ± {styled_subset.similarity.std():.3f}",
                    "Improvement": f"{improvement:+.3f}",
                    "Wilcoxon p": f"{p_val:.3f}" if not np.isnan(p_val) else "n/a"
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(results_dir / "dual_cross_artist_summary.csv", index=False)
    
    # Save cross-artist improvements as JSON for use in figure generation
    cross_artist_file = results_dir / "cross_artist_improvements.json"
    with open(cross_artist_file, 'w') as f:
        json.dump(cross_artist_improvements, f, indent=2)
    print(f"Saved cross-artist improvements to {cross_artist_file}")
    
    # Print results
    print("\n" + "="*80)
    print("DUAL EMBEDDING CROSS-ARTIST EXPERIMENT RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"   CSV files: {results_dir}/")
    print(f"   Cross-artist improvements: {cross_artist_file}")
    
    # Print cross-artist improvements matrix for verification
    print(f"\nCross-Artist Improvement Matrix:")
    for embed_type in embed_types:
        print(f"\n{embed_type.upper()}:")
        print("Generation\\Reference    Billie    Einaudi")
        for gen_artist in artists:
            row = f"{gen_artist.capitalize():15}"
            for ref_artist in artists:
                key = f"{gen_artist}_to_{ref_artist}"
                improvement = cross_artist_improvements[embed_type].get(key, 0)
                row += f"    {improvement:+.3f}"
            print(row)
    
    # Analysis summary
    for embed_type in embed_types:
        embed_df = df[df.embed_type == embed_type]
        same_artist_df = embed_df[embed_df.is_same_artist == True]
        cross_artist_df = embed_df[embed_df.is_same_artist == False]
        
        same_improvement = same_artist_df[same_artist_df.condition == 'styled'].similarity.mean() - \
                          same_artist_df[same_artist_df.condition == 'baseline'].similarity.mean()
        
        cross_improvement = cross_artist_df[cross_artist_df.condition == 'styled'].similarity.mean() - \
                           cross_artist_df[cross_artist_df.condition == 'baseline'].similarity.mean()
        
        print(f"\n{embed_type.upper()} INTERPRETATION:")
        print(f"Same-artist improvement:  {same_improvement:+.3f}")
        print(f"Cross-artist improvement: {cross_improvement:+.3f}")
        
        if same_improvement > cross_improvement:
            print(f"{embed_type.upper()} shows SPECIFICITY - better same-artist improvements!")
        else:
            print(f"WARNING: {embed_type.upper()} may lack specificity - investigate further.")

if __name__ == "__main__":
    main()

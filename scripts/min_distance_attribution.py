import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Compute minimum cosine distance from each generated clip to any reference clip"""
    min_distances = []
    
    for gen_emb in gen_embeddings:
        # Compute cosine similarity with all reference embeddings
        cos_sims = torch.nn.functional.cosine_similarity(
            gen_emb.unsqueeze(0), ref_embeddings, dim=1
        )
        # Convert to distance (1 - cosine_similarity) and take minimum
        min_dist = 1 - cos_sims.max().item()
        min_distances.append(min_dist)
    
    return np.array(min_distances)

# Main computation
results = []
all_distances = {}

for artist in ['billie', 'einaudi']:
    print(f"Computing min-distance attribution for {artist}...")
    
    all_distances[artist] = {}
    
    for embed_type in ['vggish', 'clap']:
        print(f"  {embed_type} space...")
        
        all_distances[artist][embed_type] = {}
        
        # Load reference embeddings
        ref_emb = load_embeddings(embed_type, artist, 'references')
        if ref_emb is None:
            print(f"    Missing reference embeddings for {artist} {embed_type}")
            continue
        
        # Compute distances for each condition
        conditions = ['baseline', 'artist_name_baseline', 'set1', 'set2', 'set3', 'set4', 'set5']
        
        for condition in conditions:
            gen_emb = load_embeddings(embed_type, artist, condition)
            if gen_emb is None:
                print(f"    Missing {condition} embeddings for {artist} {embed_type}")
                continue
            
            # Compute min distances
            min_dists = compute_min_distances(gen_emb, ref_emb)
            all_distances[artist][embed_type][condition] = min_dists
            
            # Store median and stats
            median_dist = np.median(min_dists)
            
            results.append({
                'artist': artist,
                'embed_type': embed_type,
                'condition': condition,
                'median_distance': median_dist,
                'mean_distance': np.mean(min_dists),
                'std_distance': np.std(min_dists)
            })

# Compute p-values (Wilcoxon tests vs baseline)
pvalue_results = []

for artist in ['billie', 'einaudi']:
    for embed_type in ['vggish', 'clap']:
        if embed_type not in all_distances[artist]:
            continue
            
        baseline_dists = all_distances[artist][embed_type].get('baseline')
        if baseline_dists is None:
            continue
        
        for condition in ['artist_name_baseline', 'set1', 'set2', 'set3', 'set4', 'set5']:
            if condition in all_distances[artist][embed_type]:
                condition_dists = all_distances[artist][embed_type][condition]
                
                # Wilcoxon test (paired, so same seeds)
                try:
                    _, p_value = wilcoxon(baseline_dists, condition_dists)
                except:
                    p_value = 1.0
                
                baseline_median = np.median(baseline_dists)
                condition_median = np.median(condition_dists)
                improvement = ((baseline_median - condition_median) / baseline_median) * 100
                
                pvalue_results.append({
                    'artist': artist,
                    'embed_type': embed_type,
                    'condition': condition,
                    'baseline_median': baseline_median,
                    'condition_median': condition_median,
                    'improvement_pct': improvement,
                    'p_value': p_value
                })

# Save results
Path("results").mkdir(exist_ok=True)
Path("figs").mkdir(exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("results/minDist_results.csv", index=False)

pvalues_df = pd.DataFrame(pvalue_results)
pvalues_df.to_csv("results/minDist_medians.csv", index=False)

# Create CDF plots
for artist in ['billie', 'einaudi']:
    for embed_type in ['vggish', 'clap']:
        if embed_type not in all_distances[artist]:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot CDFs for each condition
        conditions_to_plot = ['baseline', 'artist_name_baseline']
        # Add best performing style set
        best_condition = None
        best_median = float('inf')
        for cond in ['set1', 'set2', 'set3', 'set4', 'set5']:
            if cond in all_distances[artist][embed_type]:
                median_dist = np.median(all_distances[artist][embed_type][cond])
                if median_dist < best_median:
                    best_median = median_dist
                    best_condition = cond
        
        if best_condition:
            conditions_to_plot.append(best_condition)
        
        colors = ['red', 'blue', 'green']
        labels = ['Baseline', 'Artist Name', f'Best Style ({best_condition})']
        
        for i, condition in enumerate(conditions_to_plot):
            if condition in all_distances[artist][embed_type]:
                distances = all_distances[artist][embed_type][condition]
                
                # Compute CDF
                sorted_dists = np.sort(distances)
                p = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
                
                ax.plot(sorted_dists, p, color=colors[i], 
                       label=f"{labels[i]} (med={np.median(distances):.3f})", 
                       linewidth=2)
        
        ax.set_xlabel('Min Distance to Reference')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{artist.title()} - Min Distance CDF ({embed_type.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"figs/cdf_minDist_{embed_type}_{artist}.png", dpi=300)
        plt.close()

print("Min-distance attribution complete!")
print(f"Results saved to: results/minDist_*.csv")
print(f"CDF plots saved to: figs/cdf_minDist_*.png")

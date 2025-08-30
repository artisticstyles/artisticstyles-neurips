import tensorflow as tf
import tensorflow_hub as hub
from msclap import CLAP
from pathlib import Path
import torchaudio, torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import json
import librosa

def embed_vggish(path: Path, vggish_model):
    """Extract VGGish embeddings"""
    wav, sr = torchaudio.load(str(path))
    wav = wav.numpy().flatten()
    
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    
    wav = wav.astype(np.float32)
    embeddings = vggish_model(wav)
    embeddings = torch.from_numpy(embeddings.numpy())
    return embeddings.mean(0)

def embed_clap(path: Path, clap_model):
    """Extract CLAP embeddings"""
    audio_embed = clap_model.get_audio_embeddings([str(path)], resample=True)
    if isinstance(audio_embed, torch.Tensor):
        return audio_embed[0]
    else:
        return torch.from_numpy(audio_embed[0])

def cos_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    emb1_norm = torch.nn.functional.normalize(emb1.unsqueeze(0), dim=-1)
    emb2_norm = torch.nn.functional.normalize(emb2.unsqueeze(0), dim=-1)
    return torch.nn.functional.cosine_similarity(emb1_norm, emb2_norm).item()

def load_embeddings_from_disk(embed_type, artist, condition):
    """Load pre-computed embeddings from disk"""
    embed_dir = Path("embeddings") / embed_type / artist / condition
    if not embed_dir.exists():
        return None
    
    embeddings = []
    for npy_file in sorted(embed_dir.glob("*.npy")):
        emb = np.load(npy_file)
        embeddings.append(torch.from_numpy(emb))
    
    return torch.stack(embeddings) if embeddings else None

def main():
    print("Starting Dual Embedding Cross-Artist Experiment...")
    print("This will compute cross-artist validation for both VGGish and CLAP embeddings")
    
    # Load models
    print("Loading VGGish model...")
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
    
    print("Loading Microsoft CLAP model...")
    clap_model = CLAP(version='2023', use_cuda=False)
    
    # Create output directories
    output_dir = Path("cross-artist-outputs")
    figs_dir = output_dir / "figs"
    results_dir = output_dir / "results"
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    artists = ["billie", "einaudi"]
    embed_types = ["vggish", "clap"]
    
    all_results = []
    
    print("\nComputing cross-artist similarities for both embedding types...")
    
    for embed_type in embed_types:
        print(f"\nProcessing {embed_type.upper()} embeddings...")
        
        for gen_artist in artists:
            for ref_artist in artists:
                print(f"\n  {gen_artist.capitalize()} generation → {ref_artist.capitalize()} reference ({embed_type})")
                
                # Load reference embeddings from disk (pre-computed)
                ref_embs = load_embeddings_from_disk(embed_type, ref_artist, "references")
                if ref_embs is None:
                    print(f"    ⚠️  No reference embeddings found for {ref_artist} {embed_type}")
                    continue
                
                ref_centroid = ref_embs.mean(0, keepdim=True)
                ref_norm = torch.nn.functional.normalize(ref_centroid, dim=-1)
                
                # Load generated embeddings from disk
                baseline_embs = load_embeddings_from_disk(embed_type, gen_artist, "baseline")
                if baseline_embs is None:
                    print(f"    ⚠️  No baseline embeddings found for {gen_artist} {embed_type}")
                    continue
                
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
                
                # Also include artist_name_baseline if available
                artist_name_embs = load_embeddings_from_disk(embed_type, gen_artist, "artist_name_baseline")
                
                # Compute similarities
                def compute_sim_to_ref(embs):
                    sims = []
                    for emb in embs:
                        emb_norm = torch.nn.functional.normalize(emb.unsqueeze(0), dim=-1)
                        sim = torch.nn.functional.cosine_similarity(emb_norm, ref_norm).item()
                        sims.append(sim)
                    return sims
                
                baseline_sims = compute_sim_to_ref(baseline_embs)
                styled_sims = compute_sim_to_ref(styled_embs)
                
                # Store results for baseline
                for sim in baseline_sims:
                    all_results.append({
                        "embed_type": embed_type,
                        "generation_artist": gen_artist,
                        "reference_artist": ref_artist,
                        "condition": "baseline",
                        "is_same_artist": gen_artist == ref_artist,
                        "similarity": sim
                    })
                
                # Store results for styled
                for sim in styled_sims:
                    all_results.append({
                        "embed_type": embed_type,
                        "generation_artist": gen_artist,
                        "reference_artist": ref_artist,
                        "condition": "styled",
                        "is_same_artist": gen_artist == ref_artist,
                        "similarity": sim
                    })
                
                # Store results for artist_name if available
                if artist_name_embs is not None:
                    artist_name_sims = compute_sim_to_ref(artist_name_embs)
                    for sim in artist_name_sims:
                        all_results.append({
                            "embed_type": embed_type,
                            "generation_artist": gen_artist,
                            "reference_artist": ref_artist,
                            "condition": "artist_name",
                            "is_same_artist": gen_artist == ref_artist,
                            "similarity": sim
                        })
                
                print(f"    Baseline: {np.mean(baseline_sims):.3f} ± {np.std(baseline_sims):.3f}")
                print(f"    Styled:   {np.mean(styled_sims):.3f} ± {np.std(styled_sims):.3f}")
                if artist_name_embs is not None:
                    print(f"    Artist:   {np.mean(artist_name_sims):.3f} ± {np.std(artist_name_sims):.3f}")
    
    # Save all results
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "dual_cross_artist_similarities.csv", index=False)
    print(f"\nSaved detailed results to {results_dir / 'dual_cross_artist_similarities.csv'}")
    
    # Create summary statistics for each embedding type
    print("\nComputing summary statistics...")
    
    summary_stats = []
    cross_artist_improvements = {}  # For the heatmap data
    
    for embed_type in embed_types:
        cross_artist_improvements[embed_type] = {}
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
                
                # Store for heatmap
                key = f"{gen_artist}_to_{ref_artist}"
                cross_artist_improvements[embed_type][key] = improvement
                
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
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Dual heatmaps (VGGish and CLAP side by side)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, embed_type in enumerate(embed_types):
        embed_df = df[df.embed_type == embed_type]
        
        # Create improvement matrix
        improvement_matrix = np.zeros((2, 2))
        for gi, gen_artist in enumerate(artists):
            for ri, ref_artist in enumerate(artists):
                key = f"{gen_artist}_to_{ref_artist}"
                if key in cross_artist_improvements[embed_type]:
                    improvement_matrix[gi, ri] = cross_artist_improvements[embed_type][key]
        
        # Baseline heatmap
        baseline_pivot = embed_df[embed_df.condition == 'baseline'].groupby(
            ['generation_artist', 'reference_artist'])['similarity'].mean().unstack()
        
        im1 = axes[i, 0].imshow(baseline_pivot.values, cmap='Blues', aspect='auto')
        axes[i, 0].set_title(f'{embed_type.upper()} - Baseline Similarities')
        axes[i, 0].set_xticks([0, 1])
        axes[i, 0].set_yticks([0, 1])
        axes[i, 0].set_xticklabels(['Billie', 'Einaudi'])
        axes[i, 0].set_yticklabels(['Billie', 'Einaudi'])
        if i == 1:  # Bottom row
            axes[i, 0].set_xlabel('Reference Artist')
        axes[i, 0].set_ylabel('Generation Artist')
        
        # Add values
        for gi in range(2):
            for ri in range(2):
                axes[i, 0].text(ri, gi, f'{baseline_pivot.iloc[gi, ri]:.3f}', 
                              ha='center', va='center', fontweight='bold')
        
        # Improvement heatmap  
        im2 = axes[i, 1].imshow(improvement_matrix, cmap='RdBu_r', aspect='auto', 
                               vmin=-0.1, vmax=0.1)
        axes[i, 1].set_title(f'{embed_type.upper()} - Styled Improvements')
        axes[i, 1].set_xticks([0, 1])
        axes[i, 1].set_yticks([0, 1])
        axes[i, 1].set_xticklabels(['Billie', 'Einaudi'])
        axes[i, 1].set_yticklabels(['Billie', 'Einaudi'])
        if i == 1:  # Bottom row
            axes[i, 1].set_xlabel('Reference Artist')
        
        # Add values
        for gi in range(2):
            for ri in range(2):
                axes[i, 1].text(ri, gi, f'{improvement_matrix[gi, ri]:+.3f}', 
                              ha='center', va='center', fontweight='bold')
        
        # Add colorbar for improvements
        if i == 0:  # Top row only
            cbar = plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
            cbar.set_label('Improvement (Styled - Baseline)')
    
    plt.tight_layout()
    plt.savefig(figs_dir / "dual_cross_artist_heatmaps.png", dpi=300, bbox_inches='tight')
    
    # 2. Summary table
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Dual Embedding Cross-Artist Experiment Results', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(figs_dir / "dual_cross_artist_table.png", dpi=300, bbox_inches='tight')
    
    # Print results
    print("\n" + "="*80)
    print("DUAL EMBEDDING CROSS-ARTIST EXPERIMENT RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"   CSV files: {results_dir}/")
    print(f"   Figures: {figs_dir}/")
    print(f"   Cross-artist improvements: {cross_artist_file}")
    
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

# The Name-Free Gap: Policy-Aware Stylistic Control in Music Generation

This repository contains code for evaluating prompt-level controllability in text-to-music generation. We investigate whether lightweight, human-readable descriptors can shift generated outputs toward target artist styles with a similar impact to artist names.

## Demo & Dataset

🎵 **[Demo Website](https://artisticstyles.github.io/music-style-control-demo/)** - Listen to generated examples  
📊 **[HuggingFace Dataset](https://huggingface.co/datasets/ArtisticStyling/music-style-control-data)** - Complete reproducibility artifacts

## Quick Start

1. **Generate Music**: Use `scripts/generate_music.py` to create baseline and style token audio clips
2. **Extract Embeddings**: Run `scripts/dual_embed_extraction.py` for VGGish and CLAP embeddings  
3. **Compute Metrics**: Calculate FAD and min-distance attribution with `scripts/compute_all_style_fads.py`
4. **Cross-Artist Analysis**: Evaluate style specificity using `scripts/cross_artist_from_embeddings.py`
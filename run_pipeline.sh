#!/usr/bin/env bash
set -e

# Check for artist argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <artist_name>"
    echo "Example: $0 billie"
    exit 1
fi

ARTIST=$1

source .venv/bin/activate
python scripts/generate_music.py --artist_name $ARTIST
python scripts/embed_and_scores.py --artist_name $ARTIST
python scripts/dist_metrics.py --artist_name $ARTIST
python scripts/spectral_analysis.py --artist_name $ARTIST
python scripts/plot_box.py --artist_name $ARTIST
python scripts/create_metrics_table.py --artist_name $ARTIST

echo "Results saved to:"
echo "  results/${ARTIST}/ - CSV files and embeddings"
echo "  figs/${ARTIST}/ - All visualizations"
echo ""
echo "Key outputs:"
echo "  results/${ARTIST}/metrics_table.csv - Comprehensive metrics table"
echo "  figs/${ARTIST}/metrics_table.png - Publication-ready metrics table"
echo "  figs/${ARTIST}/metrics_breakdown.png - Detailed metrics visualization"

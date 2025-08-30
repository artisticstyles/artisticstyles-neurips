#!/usr/bin/env bash
set -e

echo "🔄 Starting Cross-Artist Experiment"
echo "This will compare:"
echo "  - Billie generation → Billie reference (same artist)"
echo "  - Billie generation → Einaudi reference (cross artist)"  
echo "  - Einaudi generation → Billie reference (cross artist)"
echo "  - Einaudi generation → Einaudi reference (same artist)"
echo ""

# Check if both artist results exist
if [ ! -d "results/billie" ] || [ ! -d "results/einaudi" ]; then
    echo "❌ Error: Need both billie and einaudi results first!"
    echo "Run these first:"
    echo "  ./run_pipeline.sh billie"
    echo "  ./run_pipeline.sh einaudi"
    exit 1
fi

source .venv/bin/activate
python scripts/cross_artist_experiment.py

echo ""
echo "✅ Cross-artist experiment complete!"
echo "Results saved to:"
echo "  📊 cross-artist-outputs/results/ - CSV files"
echo "  📈 cross-artist-outputs/figs/ - Visualizations"

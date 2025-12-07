#!/bin/bash
# Launcher script for Hamburg CE Ecosystem Streamlit Application
# Part of the AI-InnoScEnCE Project

echo "🌍 Starting Hamburg Circular Economy Ecosystem Visualizer..."
echo "Part of the AI-InnoScEnCE Project"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Activate conda environment and run Streamlit
conda run -n AI_Innoscence streamlit run app.py

# Alternative: if conda run doesn't work, use:
# eval "$(conda shell.bash hook)"
# conda activate AI_Innoscence
# streamlit run app.py

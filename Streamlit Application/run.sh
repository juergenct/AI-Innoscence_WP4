#!/bin/bash
# Convenience script to run the Hamburg CE Ecosystem Visualizer

echo "🌍 Starting Hamburg CE Ecosystem Visualizer..."
echo ""

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Check if data file exists
DATA_PATH="../Approach 2 - Scrapegraph/hamburg_ce_ecosystem/data/final/ecosystem_entities.csv"
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found at: $DATA_PATH"
    echo "   Please run the scraping pipeline first or adjust the path in app.py"
    exit 1
fi

echo "✅ Data file found"
echo "🚀 Launching Streamlit application..."
echo ""
echo "   The app will open in your browser at: http://localhost:8501"
echo ""

streamlit run app.py


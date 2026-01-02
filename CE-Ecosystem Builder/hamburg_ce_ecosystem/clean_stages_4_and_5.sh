#!/bin/bash
# Clean Stage 4 (Relationship Analysis) + Stage 5 (Graph Construction) results ONLY
# Keeps Stage 1-3 intact (verification, extraction, geocoding)

set -e

cd "$(dirname "$0")"

echo "🧹 Cleaning Stage 4 (Relationships) + Stage 5 (Graph) results..."
echo ""
echo "⚠️  NOTE: This preserves Stage 1-3 (verification, extraction, geocoding)"
echo ""

# Check current state
echo "📊 Current database state:"
echo "   Total entities: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM entity_profiles;')"
echo "   Geocoded entities: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM entity_profiles WHERE latitude IS NOT NULL;')"
echo "   Relationships: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM relationships;')"
echo "   Clusters: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM clusters;')"
echo "   Insights: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM ecosystem_insights;')"
echo "   Edges: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM edges;')"
echo ""

read -p "Proceed with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

# Clear database tables for Stage 4 and 5
echo "🗑️  Clearing database tables (relationships, clusters, ecosystem_insights, edges)..."
sqlite3 data/final/ecosystem.db <<SQL
DELETE FROM relationships;
DELETE FROM clusters;
DELETE FROM ecosystem_insights;
DELETE FROM edges;
SQL
echo "✅ Database tables cleared"
echo ""

# Clear Stage 5 output files (ecosystem_map.json, ecosystem_relationships.csv)
echo "🗑️  Removing Stage 4/5 output files..."
rm -f data/final/ecosystem_relationships.csv
rm -f data/final/ecosystem_map.json
echo "✅ Output files removed"
echo ""

# Verify cleanup
echo "✅ Verification after cleanup:"
echo "   Total entities: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM entity_profiles;')"
echo "   Geocoded entities: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM entity_profiles WHERE latitude IS NOT NULL;')"
echo "   Relationships: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM relationships;')"
echo "   Clusters: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM clusters;')"
echo "   Insights: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM ecosystem_insights;')"
echo "   Edges: $(sqlite3 data/final/ecosystem.db 'SELECT COUNT(*) FROM edges;')"
echo ""

echo "✨ Cleanup complete! Ready to run Stage 4 and 5."
echo ""
echo "🚀 Next step: Run Stages 4+5 with all 2100 geocoded entities"
echo "   cd /home/thiesen/Documents/AI-Innoscence_Ecosystem/CE-Ecosystem\ Builder"
echo "   python3 run_stages_3_and_4.py"
echo ""
echo "   OR run stages 3-5 (will skip geocoding, run relationships + graph):"
echo "   cd hamburg_ce_ecosystem"
echo "   python3 -c \"
from scrapers.batch_processor import BatchProcessor
from pathlib import Path
import json

# Load geocoded entities
profiles = json.load(open('data/extracted/entity_profiles.json'))
print(f'Loaded {len(profiles)} entities ({sum(1 for p in profiles if p.get(\"latitude\"))} geocoded)')

# Run Stage 4 + 5
processor = BatchProcessor(
    Path('data/input/entity_urls.json'),
    config_path=Path('config/scrape_config.yaml')
)
relationships, insights = processor.run_relationship_analysis(profiles)
graph = processor.build_ecosystem_graph(profiles, relationships, insights)
processor.save_results(profiles, graph, relationships)

print(f'✅ Complete: {len(relationships)} relationships, {len(graph.get(\"nodes\", []))} nodes')
\"
"

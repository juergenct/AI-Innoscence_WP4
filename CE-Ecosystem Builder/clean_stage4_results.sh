#!/bin/bash
# Clean Stage 3 (Geocoding) + Stage 4 (Relationship Analysis) results to restart with all entities

set -e

cd "$(dirname "$0")"

echo "🧹 Cleaning Stage 3 (Geocoding) + Stage 4 (Relationships) results..."
echo ""

# Backup existing results before deletion
BACKUP_DIR="hamburg_ce_ecosystem/data/relationships/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f hamburg_ce_ecosystem/data/relationships/relationships.json ]; then
    echo "📦 Backing up existing results to: $BACKUP_DIR"
    cp hamburg_ce_ecosystem/data/relationships/*.json "$BACKUP_DIR/" 2>/dev/null || true
    echo "✅ Backup complete"
    echo ""
fi

# Clear JSON files
echo "🗑️  Removing relationship JSON files..."
rm -f hamburg_ce_ecosystem/data/relationships/relationships.json
rm -f hamburg_ce_ecosystem/data/relationships/clusters.json
rm -f hamburg_ce_ecosystem/data/relationships/ecosystem_insights.json
echo "✅ JSON files removed"
echo ""

# Clear database tables
echo "🗑️  Clearing database tables (relationships, clusters, ecosystem_insights)..."
sqlite3 hamburg_ce_ecosystem/data/final/ecosystem.db <<EOF
DELETE FROM relationships;
DELETE FROM clusters;
DELETE FROM ecosystem_insights;
EOF
echo "✅ Database tables cleared"
echo ""

# Verify cleanup
echo "✅ Verification:"
echo "   Relationships: $(sqlite3 hamburg_ce_ecosystem/data/final/ecosystem.db 'SELECT COUNT(*) FROM relationships;') rows"
echo "   Clusters: $(sqlite3 hamburg_ce_ecosystem/data/final/ecosystem.db 'SELECT COUNT(*) FROM clusters;') rows"
echo "   Insights: $(sqlite3 hamburg_ce_ecosystem/data/final/ecosystem.db 'SELECT COUNT(*) FROM ecosystem_insights;') rows"
echo ""

echo "✨ Cleanup complete! Ready to process all 2100 entities."
echo ""
echo "Next step: Run Stages 3+4 script to geocode and analyze relationships"
echo "   python3 run_stages_3_and_4.py"

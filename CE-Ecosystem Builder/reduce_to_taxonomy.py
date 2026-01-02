#!/usr/bin/env python3
"""
Reduce CE Activities to Taxonomy

This script cleans up the ce_activities field in all ecosystem databases,
keeping only activities that match the predefined 120-item taxonomy.

Usage:
    python reduce_to_taxonomy.py [--dry-run]
"""

import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

# Add config path for taxonomy import
sys.path.insert(0, str(Path(__file__).parent / "hamburg_ce_ecosystem" / "config"))
from ce_activities_taxonomy import get_all_activities

# Database paths
ECOSYSTEM_DBS = [
    "hamburg_ce_ecosystem/data/final/ecosystem.db",
    "novi_sad_ce_ecosystem/data/final/ecosystem.db",
    "cahul_ce_ecosystem/data/final/ecosystem.db",
]

def reduce_activities_to_taxonomy(db_path: Path, dry_run: bool = False) -> dict:
    """
    Reduce ce_activities to only taxonomy-matching entries.

    Returns statistics about the cleanup.
    """
    taxonomy = set(get_all_activities())

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get all entities with activities
    cursor.execute("""
        SELECT url, ce_activities
        FROM entity_profiles
        WHERE ce_activities IS NOT NULL
    """)

    stats = {
        "total_entities": 0,
        "entities_with_activities": 0,
        "entities_modified": 0,
        "activities_kept": 0,
        "activities_removed": 0,
        "removed_activities": set(),
    }

    updates = []

    for url, ce_activities_str in cursor.fetchall():
        stats["total_entities"] += 1

        try:
            activities = json.loads(ce_activities_str) if ce_activities_str else []
        except json.JSONDecodeError:
            continue

        if not activities:
            continue

        stats["entities_with_activities"] += 1

        # Filter to taxonomy only
        kept = []
        removed = []

        for activity in activities:
            if activity in taxonomy:
                kept.append(activity)
            else:
                removed.append(activity)
                stats["removed_activities"].add(activity)

        stats["activities_kept"] += len(kept)
        stats["activities_removed"] += len(removed)

        # Only update if something changed
        if removed:
            stats["entities_modified"] += 1
            updates.append((json.dumps(kept), url))

    # Apply updates
    if not dry_run and updates:
        cursor.executemany("""
            UPDATE entity_profiles
            SET ce_activities = ?
            WHERE url = ?
        """, updates)
        conn.commit()

    conn.close()

    return stats


def main():
    dry_run = "--dry-run" in sys.argv

    base_path = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_path / f"taxonomy_cleanup_{timestamp}.log"

    print("=" * 60)
    print("CE Activities Taxonomy Cleanup")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (modifying databases)'}")
    print(f"Taxonomy size: {len(get_all_activities())} activities")
    print()

    all_removed = set()

    for db_rel_path in ECOSYSTEM_DBS:
        db_path = base_path / db_rel_path
        ecosystem_name = db_rel_path.split("/")[0].replace("_ce_ecosystem", "").title()

        print(f"\n--- {ecosystem_name} Ecosystem ---")

        if not db_path.exists():
            print(f"  [SKIP] Database not found: {db_path}")
            continue

        stats = reduce_activities_to_taxonomy(db_path, dry_run=dry_run)

        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Entities with activities: {stats['entities_with_activities']}")
        print(f"  Entities modified: {stats['entities_modified']}")
        print(f"  Activities kept: {stats['activities_kept']}")
        print(f"  Activities removed: {stats['activities_removed']}")

        all_removed.update(stats["removed_activities"])

    # Write log of removed activities
    print(f"\n--- Summary ---")
    print(f"Total unique activities removed across all ecosystems: {len(all_removed)}")

    with open(log_file, "w") as f:
        f.write(f"CE Activities Taxonomy Cleanup Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")
        f.write(f"\nRemoved Activities ({len(all_removed)} unique):\n")
        for activity in sorted(all_removed):
            f.write(f"  - {activity}\n")

    print(f"Log written to: {log_file}")

    if dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()

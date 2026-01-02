#!/usr/bin/env python3
"""
Filter Hamburg CE Ecosystem Entities
Removes:
1. Entities outside Hamburg metro area (>30km from center)
2. Specific non-CE entity: "Mona Li Beauty Bay"

Author: AI-InnoScEnCE Project
Date: November 11, 2025
"""

import pandas as pd
import sqlite3
import json
import math
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
HAMBURG_CENTER = (53.5511, 9.9937)  # Hamburg center coordinates
DISTANCE_THRESHOLD_KM = 30  # Hamburg metro area
MANUAL_EXCLUSIONS = ["Mona Li Beauty Bay"]  # Specific entities to remove

DATA_DIR = Path(__file__).parent / "data" / "final"
BACKUP_DIR = Path(__file__).parent / "data" / "backup_before_filtering"


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    return c * r


def create_backup():
    """Create backup of original data files"""
    print("Creating backup of original files...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "ecosystem_entities.csv",
        "ecosystem_relationships.csv",
        "ecosystem.db",
        "ecosystem_map.json"
    ]

    for filename in files_to_backup:
        src = DATA_DIR / filename
        if src.exists():
            dst = BACKUP_DIR / f"{filename}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(src, dst)
            print(f"  ✓ Backed up: {filename}")

    print(f"Backup saved to: {BACKUP_DIR}\n")


def filter_entities_csv():
    """Filter the main entities CSV file"""
    print("=" * 60)
    print("FILTERING ENTITIES CSV")
    print("=" * 60)

    csv_path = DATA_DIR / "ecosystem_entities.csv"
    df = pd.read_csv(csv_path)

    print(f"Original entity count: {len(df)}")

    # Calculate distances
    df['distance_from_hamburg'] = df.apply(
        lambda row: haversine_distance(
            HAMBURG_CENTER[0], HAMBURG_CENTER[1],
            row['latitude'], row['longitude']
        ) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else 0,
        axis=1
    )

    # Find entities to remove
    too_far = df[df['distance_from_hamburg'] > DISTANCE_THRESHOLD_KM]
    manual_exclusions_df = df[df['entity_name'].isin(MANUAL_EXCLUSIONS)]

    print(f"\nEntities outside {DISTANCE_THRESHOLD_KM}km from Hamburg: {len(too_far)}")
    if len(too_far) > 0:
        print("\nEntities to remove (geographic):")
        for _, row in too_far.iterrows():
            print(f"  - {row['entity_name']}: {row['distance_from_hamburg']:.1f} km")
            if pd.notna(row['address']):
                print(f"    Address: {row['address']}")

    print(f"\nManual exclusions: {len(manual_exclusions_df)}")
    if len(manual_exclusions_df) > 0:
        print("\nEntities to remove (manual):")
        for _, row in manual_exclusions_df.iterrows():
            print(f"  - {row['entity_name']}")

    # Combine filters
    entities_to_remove = set(too_far['entity_name'].tolist() + MANUAL_EXCLUSIONS)

    # Filter dataframe
    df_filtered = df[~df['entity_name'].isin(entities_to_remove)]

    # Drop the temporary distance column
    df_filtered = df_filtered.drop(columns=['distance_from_hamburg'])

    print(f"\nFiltered entity count: {len(df_filtered)}")
    print(f"Entities removed: {len(df) - len(df_filtered)}")

    # Save filtered CSV
    output_path = DATA_DIR / "ecosystem_entities.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"\n✓ Saved filtered entities to: {output_path}")

    return entities_to_remove


def filter_relationships_csv(removed_entities):
    """Filter relationships CSV to remove relationships involving filtered entities"""
    print("\n" + "=" * 60)
    print("FILTERING RELATIONSHIPS CSV")
    print("=" * 60)

    csv_path = DATA_DIR / "ecosystem_relationships.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping...")
        return

    df = pd.read_csv(csv_path)
    print(f"Original relationship count: {len(df)}")

    # Filter out relationships involving removed entities
    df_filtered = df[
        ~df['source_entity'].isin(removed_entities) &
        ~df['target_entity'].isin(removed_entities)
    ]

    print(f"Filtered relationship count: {len(df_filtered)}")
    print(f"Relationships removed: {len(df) - len(df_filtered)}")

    # Save filtered relationships
    df_filtered.to_csv(csv_path, index=False)
    print(f"\n✓ Saved filtered relationships to: {csv_path}")


def filter_database(removed_entities):
    """Filter SQLite database tables"""
    print("\n" + "=" * 60)
    print("FILTERING DATABASE")
    print("=" * 60)

    db_path = DATA_DIR / "ecosystem.db"

    if not db_path.exists():
        print(f"Warning: {db_path} not found, skipping...")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get entity_profiles count before
    cursor.execute("SELECT COUNT(*) FROM entity_profiles")
    before_count = cursor.fetchone()[0]
    print(f"Entity profiles before: {before_count}")

    # Delete from entity_profiles
    placeholders = ','.join('?' * len(removed_entities))
    cursor.execute(
        f"DELETE FROM entity_profiles WHERE entity_name IN ({placeholders})",
        list(removed_entities)
    )

    # Get count after
    cursor.execute("SELECT COUNT(*) FROM entity_profiles")
    after_count = cursor.fetchone()[0]
    print(f"Entity profiles after: {after_count}")
    print(f"Profiles removed: {before_count - after_count}")

    # Filter relationships table
    cursor.execute("SELECT COUNT(*) FROM relationships")
    before_rel = cursor.fetchone()[0]
    print(f"\nRelationships before: {before_rel}")

    cursor.execute(
        f"""DELETE FROM relationships
        WHERE source_entity IN ({placeholders})
           OR target_entity IN ({placeholders})""",
        list(removed_entities) * 2
    )

    cursor.execute("SELECT COUNT(*) FROM relationships")
    after_rel = cursor.fetchone()[0]
    print(f"Relationships after: {after_rel}")
    print(f"Relationships removed: {before_rel - after_rel}")

    # Commit changes
    conn.commit()
    conn.close()

    print(f"\n✓ Updated database: {db_path}")


def filter_json(removed_entities):
    """Filter ecosystem_map.json"""
    print("\n" + "=" * 60)
    print("FILTERING JSON")
    print("=" * 60)

    json_path = DATA_DIR / "ecosystem_map.json"

    if not json_path.exists():
        print(f"Warning: {json_path} not found, skipping...")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter nodes
    if 'nodes' in data:
        before = len(data['nodes'])
        data['nodes'] = [
            node for node in data['nodes']
            if node.get('id') not in removed_entities and
               node.get('entity_name') not in removed_entities
        ]
        after = len(data['nodes'])
        print(f"Nodes before: {before}")
        print(f"Nodes after: {after}")
        print(f"Nodes removed: {before - after}")

    # Filter edges
    if 'edges' in data:
        before = len(data['edges'])
        data['edges'] = [
            edge for edge in data['edges']
            if edge.get('source') not in removed_entities and
               edge.get('target') not in removed_entities
        ]
        after = len(data['edges'])
        print(f"\nEdges before: {before}")
        print(f"Edges after: {after}")
        print(f"Edges removed: {before - after}")

    # Save filtered JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved filtered JSON to: {json_path}")


def generate_report(removed_entities):
    """Generate filtering report"""
    print("\n" + "=" * 60)
    print("FILTERING REPORT")
    print("=" * 60)

    report_path = DATA_DIR / "filtering_report.txt"

    with open(report_path, 'w') as f:
        f.write("Hamburg CE Ecosystem Filtering Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Distance threshold: {DISTANCE_THRESHOLD_KM} km from Hamburg center\n")
        f.write(f"Hamburg center coordinates: {HAMBURG_CENTER}\n\n")

        f.write(f"Total entities removed: {len(removed_entities)}\n\n")

        f.write("Removed entities:\n")
        for entity in sorted(removed_entities):
            f.write(f"  - {entity}\n")

        f.write("\nFiltering criteria:\n")
        f.write(f"1. Geographic: Entities >30km from Hamburg center\n")
        f.write(f"2. Manual exclusions: {', '.join(MANUAL_EXCLUSIONS)}\n")

    print(f"✓ Report saved to: {report_path}\n")


def main():
    """Main filtering process"""
    print("\n" + "=" * 60)
    print("HAMBURG CE ECOSYSTEM ENTITY FILTER")
    print("=" * 60)
    print(f"Hamburg center: {HAMBURG_CENTER}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD_KM} km")
    print(f"Manual exclusions: {MANUAL_EXCLUSIONS}")
    print("=" * 60 + "\n")

    # Create backup
    create_backup()

    # Filter entities CSV and get list of removed entities
    removed_entities = filter_entities_csv()

    # Filter other data files
    filter_relationships_csv(removed_entities)
    filter_database(removed_entities)
    filter_json(removed_entities)

    # Generate report
    generate_report(removed_entities)

    print("\n" + "=" * 60)
    print("FILTERING COMPLETE!")
    print("=" * 60)
    print(f"✓ {len(removed_entities)} entities removed")
    print(f"✓ Backup saved to: {BACKUP_DIR}")
    print(f"✓ Report saved to: {DATA_DIR / 'filtering_report.txt'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

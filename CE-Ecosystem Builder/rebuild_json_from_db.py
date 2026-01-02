#!/usr/bin/env python3
"""Rebuild entity_profiles.json from the database."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def rebuild_entity_profiles_json(
    db_path: Path,
    output_path: Path
) -> None:
    """
    Rebuild entity_profiles.json from the database.

    Args:
        db_path: Path to ecosystem.db
        output_path: Path to output JSON file
    """
    print(f"Reading entities from database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    cur = conn.cursor()

    # Fetch all entity profiles
    cur.execute("SELECT * FROM entity_profiles;")
    rows = cur.fetchall()

    print(f"Found {len(rows)} entities in database")

    # Convert to list of dicts
    entities = []
    for row in rows:
        entity = dict(row)

        # Parse JSON fields
        json_fields = [
            'contact_persons', 'emails', 'phone_numbers', 'ce_activities',
            'capabilities_offered', 'needs_requirements', 'capability_categories',
            'partners', 'partner_urls', 'ce_activities_structured',
            'ce_capabilities_offered', 'ce_needs_requirements',
            'mentioned_partners', 'discovered_entities'
        ]

        for field in json_fields:
            if entity.get(field):
                try:
                    entity[field] = json.loads(entity[field])
                except (json.JSONDecodeError, TypeError):
                    entity[field] = []
            else:
                entity[field] = []

        # Remove database-specific fields
        entity.pop('id', None)

        entities.append(entity)

    conn.close()

    # Save to JSON file
    print(f"Writing {len(entities)} entities to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        print(f"Backing up existing file to: {backup_path}")
        output_path.rename(backup_path)

    output_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"✅ Successfully rebuilt entity_profiles.json with {len(entities)} entities")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent / 'hamburg_ce_ecosystem'
    db_path = base_dir / 'data' / 'final' / 'ecosystem.db'
    output_path = base_dir / 'data' / 'extracted' / 'entity_profiles.json'

    rebuild_entity_profiles_json(db_path, output_path)

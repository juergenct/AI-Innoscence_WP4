from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Dict, Any
from enum import Enum


def _enum_value(val: Any) -> str:
    if isinstance(val, Enum):
        return val.value
    return str(val) if val is not None else ""


def _list_of_str(items: Any) -> list[str]:
    if not items:
        return []
    return [str(x) for x in items]


class DataManager:
    def __init__(self, db_path: str | Path = None):
        default_path = Path(__file__).resolve().parents[2] / 'data' / 'final' / 'ecosystem.db'
        self.db_path = Path(db_path) if db_path else default_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute('PRAGMA journal_mode=WAL;')
        self._conn.execute('PRAGMA synchronous=NORMAL;')
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS verification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                is_hamburg_based INTEGER,
                hamburg_confidence REAL,
                is_ce_related INTEGER,
                ce_confidence REAL,
                reasoning TEXT,
                should_extract INTEGER,
                input_category TEXT
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS entity_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                entity_name TEXT,
                ecosystem_role TEXT,
                contact_persons TEXT,
                emails TEXT,
                phone_numbers TEXT,
                brief_description TEXT,
                ce_relation TEXT,
                ce_activities TEXT,
                capabilities_offered TEXT,
                needs_requirements TEXT,
                capability_categories TEXT,
                partners TEXT,
                partner_urls TEXT,
                ce_activities_structured TEXT,
                ce_capabilities_offered TEXT,
                ce_needs_requirements TEXT,
                mentioned_partners TEXT,
                discovered_entities TEXT,
                address TEXT,
                latitude REAL,
                longitude REAL,
                extraction_timestamp TEXT,
                extraction_confidence REAL,
                raw_html_path TEXT,
                raw_html_original_path TEXT
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                type TEXT
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                target_entity TEXT,
                relationship_type TEXT,
                confidence REAL,
                evidence TEXT,
                bidirectional INTEGER,
                source_url TEXT,
                target_url TEXT,
                discovery_chain TEXT,
                matching_confidence REAL,
                UNIQUE(source_entity, target_entity, relationship_type)
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT UNIQUE,
                cluster_name TEXT,
                cluster_type TEXT,
                description TEXT,
                entities TEXT,
                items TEXT,
                confidence REAL
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS discovered_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovered_entity_url TEXT,
                discovered_entity_name TEXT,
                discovered_by_entity TEXT,
                discovered_by_url TEXT,
                discovery_depth INTEGER,
                discovery_chain TEXT,
                timestamp TEXT,
                processed INTEGER,
                UNIQUE(discovered_entity_url, discovered_by_url)
            );'''
        )
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS ecosystem_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                title TEXT,
                description TEXT,
                entities_involved TEXT,
                confidence REAL,
                priority TEXT,
                timestamp TEXT
            );'''
        )

        # Create indexes for performance optimization
        # Relationships table indexes (most frequently queried)
        cur.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_relationships_confidence ON relationships(confidence);')

        # Entity profiles indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_entity_profiles_name ON entity_profiles(entity_name);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_entity_profiles_role ON entity_profiles(ecosystem_role);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_entity_profiles_confidence ON entity_profiles(extraction_confidence);')

        # Verification results indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_verification_url ON verification_results(url);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_verification_hamburg ON verification_results(is_hamburg_based);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_verification_ce ON verification_results(is_ce_related);')

        # Ecosystem insights indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_insights_type ON ecosystem_insights(insight_type);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_insights_priority ON ecosystem_insights(priority);')

        # Clusters indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_clusters_type ON clusters(cluster_type);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_clusters_id ON clusters(cluster_id);')

        # Discovered entities indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_discovered_url ON discovered_entities(discovered_entity_url);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_discovered_processed ON discovered_entities(processed);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_discovered_depth ON discovered_entities(discovery_depth);')

        self._conn.commit()

    def save_verification(self, items: Iterable[Dict[str, Any]]) -> None:
        cur = self._conn.cursor()
        for it in items:
            cur.execute(
                '''INSERT INTO verification_results
                (url, is_hamburg_based, hamburg_confidence, is_ce_related, ce_confidence, reasoning, should_extract, input_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    is_hamburg_based=excluded.is_hamburg_based,
                    hamburg_confidence=excluded.hamburg_confidence,
                    is_ce_related=excluded.is_ce_related,
                    ce_confidence=excluded.ce_confidence,
                    reasoning=excluded.reasoning,
                    should_extract=excluded.should_extract,
                    input_category=excluded.input_category
                ;''', (
                    str(it.get('url') or ""),
                    int(bool(it.get('is_hamburg_based'))),
                    float(it.get('hamburg_confidence') or 0.0),
                    int(bool(it.get('is_ce_related'))),
                    float(it.get('ce_confidence') or 0.0),
                    it.get('verification_reasoning') or '',
                    int(bool(it.get('should_extract'))),
                    str(it.get('input_category') or ''),
                )
            )
        self._conn.commit()

    def save_profiles(self, items: Iterable[Dict[str, Any]]) -> None:
        cur = self._conn.cursor()
        for it in items:
            cur.execute(
                '''INSERT INTO entity_profiles
                (url, entity_name, ecosystem_role, contact_persons, emails, phone_numbers, brief_description, ce_relation, ce_activities, capabilities_offered, needs_requirements, capability_categories, partners, partner_urls, ce_activities_structured, ce_capabilities_offered, ce_needs_requirements, mentioned_partners, discovered_entities, address, latitude, longitude, extraction_timestamp, extraction_confidence, raw_html_path, raw_html_original_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    entity_name=excluded.entity_name,
                    ecosystem_role=excluded.ecosystem_role,
                    contact_persons=excluded.contact_persons,
                    emails=excluded.emails,
                    phone_numbers=excluded.phone_numbers,
                    brief_description=excluded.brief_description,
                    ce_relation=excluded.ce_relation,
                    ce_activities=excluded.ce_activities,
                    capabilities_offered=excluded.capabilities_offered,
                    needs_requirements=excluded.needs_requirements,
                    capability_categories=excluded.capability_categories,
                    partners=excluded.partners,
                    partner_urls=excluded.partner_urls,
                    ce_activities_structured=excluded.ce_activities_structured,
                    ce_capabilities_offered=excluded.ce_capabilities_offered,
                    ce_needs_requirements=excluded.ce_needs_requirements,
                    mentioned_partners=excluded.mentioned_partners,
                    discovered_entities=excluded.discovered_entities,
                    address=excluded.address,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    extraction_timestamp=excluded.extraction_timestamp,
                    extraction_confidence=excluded.extraction_confidence,
                    raw_html_path=excluded.raw_html_path,
                    raw_html_original_path=excluded.raw_html_original_path
                ;''', (
                    str(it.get('url') or ""),
                    it.get('entity_name') or '',
                    _enum_value(it.get('ecosystem_role')),
                    json.dumps(_list_of_str(it.get('contact_persons')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('emails')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('phone_numbers')), ensure_ascii=False),
                    it.get('brief_description') or '',
                    it.get('ce_relation') or '',
                    json.dumps(_list_of_str(it.get('ce_activities')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('capabilities_offered')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('needs_requirements')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('capability_categories')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('partners')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('partner_urls')), ensure_ascii=False),
                    json.dumps(it.get('ce_activities_structured') or [], ensure_ascii=False),
                    json.dumps(it.get('ce_capabilities_offered') or [], ensure_ascii=False),
                    json.dumps(it.get('ce_needs_requirements') or [], ensure_ascii=False),
                    json.dumps(it.get('mentioned_partners') or [], ensure_ascii=False),
                    json.dumps(it.get('discovered_entities') or [], ensure_ascii=False),
                    it.get('address') or '',
                    it.get('latitude'),
                    it.get('longitude'),
                    it.get('extraction_timestamp') or '',
                    float(it.get('extraction_confidence') or 0.0),
                    it.get('raw_html_path') or '',
                    it.get('raw_html_original_path') or '',
                )
            )
        self._conn.commit()

    def save_edges(self, edges: Iterable[Dict[str, Any]]) -> None:
        cur = self._conn.cursor()
        cur.executemany(
            '''INSERT INTO edges (source, target, type) VALUES (?, ?, ?);''',
            [(
                str(e.get('source') or ''),
                str(e.get('target') or ''),
                str(e.get('type') or ''),
            ) for e in edges]
        )
        self._conn.commit()

    def save_relationships(self, relationships: Iterable[Dict[str, Any]]) -> None:
        """Save relationship analysis results to database."""
        cur = self._conn.cursor()
        for rel in relationships:
            cur.execute(
                '''INSERT INTO relationships
                (source_entity, target_entity, relationship_type, confidence, evidence, bidirectional, source_url, target_url, discovery_chain, matching_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity, target_entity, relationship_type) DO UPDATE SET
                    confidence=excluded.confidence,
                    evidence=excluded.evidence,
                    bidirectional=excluded.bidirectional,
                    source_url=excluded.source_url,
                    target_url=excluded.target_url,
                    discovery_chain=excluded.discovery_chain,
                    matching_confidence=excluded.matching_confidence
                ;''', (
                    str(rel.get('source_entity') or ''),
                    str(rel.get('target_entity') or ''),
                    str(rel.get('relationship_type') or ''),
                    float(rel.get('confidence') or 0.0),
                    str(rel.get('evidence') or ''),
                    int(bool(rel.get('bidirectional', False))),
                    str(rel.get('source_url') or ''),
                    str(rel.get('target_url') or ''),
                    str(rel.get('discovery_chain') or ''),
                    float(rel.get('matching_confidence') or 0.0) if rel.get('matching_confidence') is not None else None,
                )
            )
        self._conn.commit()

    def save_clusters(self, clusters: Iterable[Dict[str, Any]]) -> None:
        """Save cluster analysis results to database."""
        cur = self._conn.cursor()
        for cluster in clusters:
            cur.execute(
                '''INSERT INTO clusters
                (cluster_id, cluster_name, cluster_type, description, entities, items, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cluster_id) DO UPDATE SET
                    cluster_name=excluded.cluster_name,
                    cluster_type=excluded.cluster_type,
                    description=excluded.description,
                    entities=excluded.entities,
                    items=excluded.items,
                    confidence=excluded.confidence
                ;''', (
                    str(cluster.get('cluster_id') or ''),
                    str(cluster.get('cluster_name') or ''),
                    str(cluster.get('cluster_type') or ''),
                    str(cluster.get('description') or ''),
                    json.dumps(_list_of_str(cluster.get('entities')), ensure_ascii=False),
                    json.dumps(_list_of_str(cluster.get('items')), ensure_ascii=False),
                    float(cluster.get('confidence') or 0.0),
                )
            )
        self._conn.commit()

    def save_discovered_entities(self, discovered: Iterable[Dict[str, Any]]) -> None:
        """Save discovered entity records to database."""
        cur = self._conn.cursor()
        for record in discovered:
            cur.execute(
                '''INSERT INTO discovered_entities
                (discovered_entity_url, discovered_entity_name, discovered_by_entity, discovered_by_url, discovery_depth, discovery_chain, timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(discovered_entity_url, discovered_by_url) DO UPDATE SET
                    discovered_entity_name=excluded.discovered_entity_name,
                    discovery_depth=excluded.discovery_depth,
                    discovery_chain=excluded.discovery_chain,
                    timestamp=excluded.timestamp,
                    processed=excluded.processed
                ;''', (
                    str(record.get('discovered_entity_url') or ''),
                    str(record.get('discovered_entity_name') or ''),
                    str(record.get('discovered_by_entity') or ''),
                    str(record.get('discovered_by_url') or ''),
                    int(record.get('discovery_depth') or 0),
                    str(record.get('discovery_chain') or ''),
                    str(record.get('timestamp') or ''),
                    int(bool(record.get('processed', False))),
                )
            )
        self._conn.commit()

    def save_insights(self, insights: Iterable[Dict[str, Any]]) -> None:
        """Save ecosystem insights to database."""
        cur = self._conn.cursor()
        for insight in insights:
            entities_involved = insight.get('entities_involved', [])
            entities_json = json.dumps(_list_of_str(entities_involved), ensure_ascii=False)

            cur.execute(
                '''INSERT INTO ecosystem_insights
                (insight_type, title, description, entities_involved, confidence, priority, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?);''', (
                    str(insight.get('insight_type') or ''),
                    str(insight.get('title') or ''),
                    str(insight.get('description') or ''),
                    entities_json,
                    float(insight.get('confidence') or 0.0),
                    str(insight.get('priority') or 'medium'),
                    str(insight.get('timestamp') or ''),
                )
            )
        self._conn.commit()

    def get_extracted_urls(self) -> set[str]:
        """Get set of URLs that have already been extracted and saved to entity_profiles table.

        Returns:
            Set of URLs that have already been successfully extracted.
        """
        cur = self._conn.cursor()
        cur.execute('SELECT url FROM entity_profiles;')
        return {row[0] for row in cur.fetchall()}

    def get_entity_count(self) -> int:
        """Get total count of entities in entity_profiles table.

        Returns:
            Total number of entity profiles in database.
        """
        cur = self._conn.cursor()
        cur.execute('SELECT COUNT(*) FROM entity_profiles;')
        return cur.fetchone()[0]

    def get_geocoded_urls(self) -> set[str]:
        """Get set of URLs that have already been geocoded (have coordinates).

        Returns:
            Set of URLs that have latitude and longitude values.
        """
        cur = self._conn.cursor()
        cur.execute('SELECT url FROM entity_profiles WHERE latitude IS NOT NULL AND longitude IS NOT NULL;')
        return {row[0] for row in cur.fetchall()}

    def get_all_entity_profiles(self) -> list[Dict[str, Any]]:
        """Get all entity profiles from database as list of dicts.

        Returns:
            List of entity profile dictionaries with parsed JSON fields.
        """
        cur = self._conn.cursor()
        cur.execute('SELECT * FROM entity_profiles;')

        # Get column names
        columns = [description[0] for description in cur.description]

        entities = []
        for row in cur.fetchall():
            entity = dict(zip(columns, row))

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

            # Remove database-specific id field
            entity.pop('id', None)
            entities.append(entity)

        return entities

    def update_coordinates(self, url: str, latitude: float, longitude: float) -> None:
        """Update geocoded coordinates for a single entity.

        Args:
            url: Entity URL to update
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        """
        cur = self._conn.cursor()
        cur.execute(
            '''UPDATE entity_profiles
            SET latitude = ?, longitude = ?
            WHERE url = ?;''',
            (latitude, longitude, str(url))
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

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
                partners TEXT,
                partner_urls TEXT,
                address TEXT,
                latitude REAL,
                longitude REAL,
                extraction_timestamp TEXT,
                extraction_confidence REAL
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
                (url, entity_name, ecosystem_role, contact_persons, emails, phone_numbers, brief_description, ce_relation, ce_activities, partners, partner_urls, address, latitude, longitude, extraction_timestamp, extraction_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    entity_name=excluded.entity_name,
                    ecosystem_role=excluded.ecosystem_role,
                    contact_persons=excluded.contact_persons,
                    emails=excluded.emails,
                    phone_numbers=excluded.phone_numbers,
                    brief_description=excluded.brief_description,
                    ce_relation=excluded.ce_relation,
                    ce_activities=excluded.ce_activities,
                    partners=excluded.partners,
                    partner_urls=excluded.partner_urls,
                    address=excluded.address,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    extraction_timestamp=excluded.extraction_timestamp,
                    extraction_confidence=excluded.extraction_confidence
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
                    json.dumps(_list_of_str(it.get('partners')), ensure_ascii=False),
                    json.dumps(_list_of_str(it.get('partner_urls')), ensure_ascii=False),
                    it.get('address') or '',
                    it.get('latitude'),
                    it.get('longitude'),
                    it.get('extraction_timestamp') or '',
                    float(it.get('extraction_confidence') or 0.0),
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

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

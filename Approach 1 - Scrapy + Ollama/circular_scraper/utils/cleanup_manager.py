"""
Cleanup Manager

Removes legacy/unused data while keeping recent sessions and key exports.

Defaults:
- Keep last 2 session folders under data/raw and data/processed
- Keep exports for those sessions
- Keep last 2 generations of iteration/link stats in data/exports

Usage:
  python -m circular_scraper.utils.cleanup_manager \
    --data-dir data \
    --keep-sessions 2 \
    --keep-generations 2 \
    --dry-run false
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


SESSION_PATTERN = re.compile(r"^(\d{8}_\d{6})$")
FILE_SESSION_PATTERN = re.compile(r"^(?:entities|links|errors)_(\d{8}_\d{6})_\d{4}\.(?:csv|json|parquet)$")


def list_session_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and SESSION_PATTERN.match(p.name)]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def keep_latest(paths: List[Path], keep: int) -> Tuple[List[Path], List[Path]]:
    keep_paths = paths[:keep]
    delete_paths = paths[keep:]
    return keep_paths, delete_paths


def delete_paths(paths: List[Path], dry_run: bool):
    for p in paths:
        if dry_run:
            print(f"DRY-RUN: would delete {p}")
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def cleanup_sessions(data_dir: Path, keep_sessions: int, dry_run: bool) -> List[str]:
    raw = data_dir / 'raw'
    processed = data_dir / 'processed'
    raw_sessions = list_session_dirs(raw)
    proc_sessions = list_session_dirs(processed)
    kept_session_names: List[str] = []

    # Decide sessions to keep based on raw and processed combined (union)
    combined = sorted({p.name: p for p in raw_sessions + proc_sessions}.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    keep_paths, delete_paths = keep_latest(combined, keep_sessions)
    kept_session_names = [p.name for p in keep_paths]

    # Delete in raw/processed any session not kept
    for base in [raw, processed]:
        sessions = list_session_dirs(base)
        for s in sessions:
            if s.name not in kept_session_names:
                delete_paths([s], dry_run)

    return kept_session_names


def cleanup_exports(data_dir: Path, kept_sessions: List[str], keep_generations: int, dry_run: bool):
    exports = data_dir / 'exports'
    if not exports.exists():
        return

    # Subdirs csv/parquet/json
    for sub in ['csv', 'parquet', 'json']:
        d = exports / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            m = FILE_SESSION_PATTERN.match(f.name)
            if m:
                sess = m.group(1)
                if sess not in kept_sessions:
                    delete_paths([f], dry_run)

    # Root patterns: discovered_links_all_*.txt, iteration_seeds_*.{csv,txt}, link_stats_*.json, spider_stats_*.json, summary_*.json, llm_classifications_*.json
    patterns_keep_n = [
        ('discovered_links_all_*.txt', keep_generations),
        ('iteration_seeds_*.csv', keep_generations),
        ('iteration_seeds_*.txt', keep_generations),
        ('link_stats_*.json', keep_generations),
        ('spider_stats_*.json', keep_generations),
        ('summary_*.json', keep_generations),
        ('llm_classifications_*.json', keep_generations),
        ('circular_economy_data_*.xlsx', keep_generations),
    ]
    for pattern, keep_n in patterns_keep_n:
        files = sorted(exports.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        _, del_files = keep_latest(files, keep_n)
        delete_paths(del_files, dry_run)


def cleanup_test_files(project_root: Path, dry_run: bool):
    # Remove ad-hoc test seed files created during testing
    for pattern in ['test_*.csv', 'test_*.sh', '*_TEST_*.sh']:
        for f in project_root.glob(pattern):
            delete_paths([f], dry_run)


def main():
    parser = argparse.ArgumentParser(description='Cleanup legacy/unused data')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--keep-sessions', type=int, default=2, help='How many recent session folders to keep')
    parser.add_argument('--keep-generations', type=int, default=2, help='How many generations of top-level exports to keep')
    parser.add_argument('--dry-run', type=str, default='false', help='If true, only print what would be deleted')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    project_root = Path(__file__).resolve().parents[3]
    dry_run = str(args.dry_run).lower() == 'true'

    kept_sessions = cleanup_sessions(data_dir, args.keep_sessions, dry_run)
    cleanup_exports(data_dir, kept_sessions, args.keep_generations, dry_run)
    cleanup_test_files(project_root, dry_run)

    print(f"Kept sessions: {kept_sessions}")
    print("Cleanup complete" + (" (dry-run)" if dry_run else ""))


if __name__ == '__main__':
    main()



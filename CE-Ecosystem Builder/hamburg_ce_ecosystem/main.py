from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor


def prepare_input_urls(base_dir: Path) -> Path:
    # Check if preprocessed entity_urls.json exists first
    data_input = base_dir / 'data' / 'input'
    data_input.mkdir(parents=True, exist_ok=True)
    out_path = data_input / 'entity_urls.json'
    
    # If preprocessed data exists, use it
    if out_path.exists():
        return out_path
    
    # Otherwise, fall back to config/entity_sources.json
    config_sources = base_dir / 'config' / 'entity_sources.json'
    if config_sources.exists():
        out_path.write_text(config_sources.read_text(encoding='utf-8'), encoding='utf-8')
        return out_path

    # Final fallback: create minimal sample
    urls = {
        "Students": [
            "https://www.tuhh.de/tune",
            "https://enactus-hamburg.de/",
        ],
        "Researchers": [],
        "Industry Partners": [],
    }
    out_path.write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    print("Hamburg CE Ecosystem Mapper")
    print("-" * 50)

    # Ensure Ollama is running
    try:
        subprocess.run(['ollama', 'list'], check=True, capture_output=True)
    except Exception:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        return

    input_file = prepare_input_urls(base_dir)
    processor = BatchProcessor(input_file, config_path=base_dir / 'config' / 'scrape_config.yaml')
    processor.run_pipeline()
    print("Pipeline complete! Results in data/final/")


if __name__ == '__main__':
    main()

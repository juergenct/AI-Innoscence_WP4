# Circular Economy Hamburg Web Scraper - Comprehensive Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Workflow Pipeline](#workflow-pipeline)
4. [Core Components](#core-components)
5. [Data Models](#data-models)
6. [Spiders](#spiders)
7. [Middlewares](#middlewares)
8. [Pipelines](#pipelines)
9. [Utilities](#utilities)
10. [Iteration System](#iteration-system)
11. [LLM Classification](#llm-classification)
12. [Data Storage Structure](#data-storage-structure)
13. [Configuration & Settings](#configuration--settings)
14. [Running & Operations](#running--operations)
15. [Troubleshooting](#troubleshooting)

---

## Overview

The Circular Economy Hamburg Web Scraper is a sophisticated Scrapy-based system designed to discover, crawl, and classify organizations related to the circular economy ecosystem in Hamburg, Germany. It implements an iterative discovery approach with intelligent filtering and LLM-based classification.

### Key Features
- **Dual-mode crawling**: Static HTML and dynamic JavaScript rendering
- **Entity resolution**: Smart grouping of multi-page organizations
- **Hamburg-focused**: Multi-strategy Hamburg relevance detection
- **CE detection**: Comprehensive circular economy keyword matching
- **LLM classification**: 14 ecosystem roles via local Ollama models
- **Iterative discovery**: Multi-round crawling with intelligent link filtering
- **Entity-organized storage**: Hierarchical data organization by entity

### Technology Stack
- **Core**: Python 3.8+, Scrapy 2.x
- **JavaScript rendering**: Playwright via scrapy-playwright
- **Text extraction**: trafilatura, BeautifulSoup4
- **Data processing**: pandas, pyarrow
- **LLM**: Ollama with Qwen/Qwen3-4B-Instruct-2507
- **Storage**: CSV, Parquet, JSON, HTML, TXT

---

## System Architecture

```
┌─────────────────────┐
│   Seed URLs (CSV)   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐     ┌────────────────┐
│   Spider Engine     │────▶│  Middlewares   │
│  (Static/Dynamic)   │     │  (User-Agent,  │
└──────────┬──────────┘     │   Playwright)  │
           ▼                └────────────────┘
┌─────────────────────┐
│  Entity Resolution  │
│  & Hamburg Check    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Item Pipelines    │
│  ┌──────────────┐   │
│  │ Validation   │   │
│  ├──────────────┤   │
│  │ Text Extract │   │
│  ├──────────────┤   │
│  │ Data Storage │   │
│  ├──────────────┤   │
│  │ Link Extract │   │
│  └──────────────┘   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐     ┌────────────────┐
│   Data Storage      │     │ LLM Classifier │
│  (Entity-Organized) │────▶│   (Ollama)     │
└──────────┬──────────┘     └────────────────┘
           ▼
┌─────────────────────┐
│  Iteration Seeds    │
│  (Filtered Links)   │
└─────────────────────┘
```

---

## Workflow Pipeline

### Complete Iterative Discovery Process

1. **Initialization Phase**
   - Load seed URLs from CSV (columns: website/url/URL)
   - Normalize URLs (add HTTPS, deduplicate)
   - Initialize entity resolver and relevance tracking

2. **Crawling Phase** (Per URL)
   - Spider selection (Static vs Dynamic based on domain)
   - Request with depth tracking and parent URL
   - Middleware processing (User-Agent rotation, Playwright upgrade)
   - Response parsing and item creation

3. **Entity Resolution Phase**
   - Group pages by entity (domain or subdomain for academic sites)
   - Assign stable entity_id, entity_name, entity_root_url
   - Track entity relevance for crawling decisions

4. **Hamburg Detection Phase** (6 strategies)
   - URL/domain patterns (hamburg, hh.de, hansestadt)
   - Text keywords (official references, institution names)
   - Postal codes (20000-22999)
   - District names (7 Bezirke + Stadtteile)
   - Hamburg institutions (universities, landmarks)
   - Outbound link analysis

5. **Content Extraction Phase**
   - Primary: trafilatura for high-quality text extraction
   - Fallback: BeautifulSoup for HTML parsing
   - Metadata extraction (title, description, language)
   - Contact extraction (emails, phones, social media)
   - CE keyword detection (recycling, sustainability, etc.)

6. **Data Storage Phase** (Entity-organized)
   - Raw HTML with metadata JSON
   - Processed text with headers
   - Entity summary aggregation
   - Buffered writes to CSV/Parquet/JSON

7. **Link Discovery Phase**
   - Extract all internal and external links
   - Track which entity discovered each link
   - Identify potential new entities
   - Save ALL links for reference

8. **Filtering Phase** (Hamburg+CE)
   - Filter links to only those from relevant entities
   - Create iteration seeds CSV for next round
   - Generate statistics report

9. **LLM Classification Phase**
   - Sample pages per entity (default 3)
   - Classify Hamburg relevance, CE relevance, ecosystem role
   - 14 specific roles with definitions
   - Merge results into aggregated data

10. **Iteration Loop**
    - Use filtered seeds for next crawl
    - Entity deduplication (one root URL per entity)
    - Repeat for N iterations (default 4-5)
    - Stop when no new relevant entities found

---

## Core Components

### Project Structure
```
AI-Innoscence_Ecosystem/
├── circular_scraper/          # Main package
│   ├── __init__.py
│   ├── items.py              # Data models
│   ├── middlewares.py        # Request/response processing
│   ├── pipelines.py          # Data processing pipelines
│   ├── settings.py           # Scrapy configuration
│   ├── spiders/              # Spider implementations
│   │   ├── base_spider.py    # Base spider class
│   │   ├── static_spider.py  # Static HTML spider
│   │   └── dynamic_spider.py # JavaScript-enabled spider
│   └── utils/                # Utility modules
│       ├── entity_resolver.py # Entity grouping logic
│       ├── export_manager.py  # Data export utilities
│       ├── llm_classifier.py  # LLM classification
│       └── url_manager.py     # URL normalization
├── data/                      # Data storage
│   ├── raw/                  # Original HTML
│   ├── processed/            # Extracted text
│   └── exports/              # Structured data exports
├── docs/                     # Documentation
├── run_scraper.py           # Main entry point
├── requirements.txt         # Python dependencies
└── scrapy.cfg              # Scrapy project config
```

---

## Data Models

### CircularEconomyItem
Primary data container for scraped pages.

**Metadata Fields:**
- `url`: Page URL
- `domain`: Domain name
- `scraped_at`: ISO timestamp
- `crawl_depth`: Distance from seed
- `parent_url`: Referrer URL

**Entity Resolution Fields:**
- `entity_id`: Stable entity identifier
- `entity_name`: Human-readable name
- `entity_root_url`: Entity root for scoping

**Page Information Fields:**
- `title`: Page title
- `meta_description`: Meta description
- `language`: Language code (de/en/un)
- `page_type`: 'static' or 'dynamic'

**Content Fields:**
- `raw_html`: Complete HTML
- `extracted_text`: Extracted plain text
- `main_content`: Main content area
- `structured_data`: JSON-LD if present

**Links & Contacts Fields:**
- `internal_links`: Same-domain links
- `external_links`: Cross-domain links
- `emails`: Extracted emails
- `phone_numbers`: Extracted phones
- `social_media`: Social media profiles

**Classification Fields:**
- `has_circular_economy_terms`: CE relevance
- `has_hamburg_reference`: Hamburg relevance
- `keywords`: Found CE keywords

---

## Spiders

### BaseCircularEconomySpider
Abstract base class providing common functionality.

**Key Features:**
- URL loading from CSV or command line
- Entity resolution and relevance tracking
- Hamburg detection (enhanced 6-strategy approach)
- Depth-limited crawling
- Statistics collection

**Entity Relevance Gating:**
- Initial discovery at depth 0: crawl all
- Depth > 0: only follow if entity is Hamburg-relevant
- Candidate pages (contact/about) always checked

### StaticSpider
For standard HTML websites.
- Faster delays (1.5s)
- Higher concurrency (3 per domain)
- Domain-specific parsers for universities, corporations, government

### DynamicSpider
For JavaScript-heavy applications.
- Playwright integration
- Wait strategies (networkidle, selectors)
- Cookie banner auto-acceptance

---

## Middlewares

### RotateUserAgentMiddleware
Randomizes User-Agent headers to reduce bot detection.

### SmartRenderMiddleware
Auto-detects JavaScript requirements and enables Playwright when needed.

### DepthLimitMiddleware
Optional per-domain depth limits to prevent infinite crawls.

---

## Pipelines

### 1. ValidationPipeline (Priority: 100)
- URL deduplication
- Required field validation
- Default value assignment

### 2. TextExtractionPipeline (Priority: 200)
- Primary: trafilatura extraction
- Fallback: BeautifulSoup
- Language detection
- Contact extraction
- Keyword detection

### 3. DataStoragePipeline (Priority: 300)
- Entity-organized storage
- Buffered writes
- Multiple formats (CSV, Parquet, JSON)
- Entity summary aggregation

### 4. LinkExtractionPipeline (Priority: 400)
- Extract all links
- Track source entity
- Dual output: ALL links + Filtered seeds
- Statistics generation

---

## Utilities

### EntityResolver
Groups pages into logical entities.
- Default: Group by domain
- Academic: Group by first path segment
- Multi-entity domain support

### ExportManager
Data consolidation and export.
- Session data merging
- Multi-format export
- Summary generation

### LLMClassifier
Entity classification via Ollama.
- 14 ecosystem roles
- Hamburg/CE relevance
- Confidence scoring

### URLManager
URL normalization and tracking.
- Query parameter sorting
- Tracker removal
- Domain statistics

---

## Iteration System

### Workflow
1. Initial crawl with seeds
2. Discover all links
3. Filter Hamburg+CE relevant
4. Entity deduplication
5. Next iteration with filtered seeds

### Output Files
- `discovered_links_all_*.txt`: All discovered links
- `iteration_seeds_*.csv`: Filtered for next iteration
- `link_stats_*.json`: Statistics
- `entities_aggregated.csv`: Entity summary

---

## LLM Classification

### 14 Ecosystem Roles
1. Students
2. Researchers
3. Higher Education Institutions
4. Research Institutes
5. Non-Governmental Organizations
6. Industry Partners
7. Startups and Entrepreneurs
8. Public Authorities
9. Policy Makers
10. End-Users
11. Citizen Associations
12. Media and Communication Partners
13. Funding Bodies
14. Knowledge and Innovation Communities

### Classification Process
1. Group by entity_id
2. Sample pages (default 3)
3. Query Ollama
4. Parse JSON response
5. Merge results

---

## Data Storage Structure

```
data/
├── raw/SESSION_ID/entities/EntityName_entity_id/
│   ├── PageTitle_hash.html
│   └── PageTitle_hash_meta.json
├── processed/SESSION_ID/entities/EntityName_entity_id/
│   ├── PageTitle_hash.txt
│   └── _entity_summary.json
└── exports/
    ├── csv/entities_*.csv
    ├── discovered_links_all_*.txt
    ├── iteration_seeds_*.csv
    └── entities_aggregated.csv
```

---

## Configuration & Settings

### Key Settings
```python
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_DELAY = 2
DEPTH_LIMIT = 3
DOWNLOAD_TIMEOUT = 30
MEMUSAGE_LIMIT_MB = 2048
```

### Setup
```bash
pip install -r requirements.txt
playwright install chromium
ollama pull qwen3-4b-instruct
```

---

## Running & Operations

### Basic Commands
```bash
# Single URL
python run_scraper.py --url https://example.com

# CSV seeds
python run_scraper.py --seed stakeholders.csv

# Iterative discovery
python run_scraper.py --seed initial.csv --iterate 5

# With classification
python run_scraper.py --seed initial.csv --classify
```

### Command Line Arguments
- `--seed`: CSV file with URLs
- `--url`: Single URL
- `--depth`: Max crawl depth (default 3)
- `--iterate`: Number of iterations
- `--classify`: Run LLM classification
- `--export`: Export format

---

## Troubleshooting

### Common Issues

**Reactor Error**: Ensure asyncio reactor in settings
**Playwright Issues**: Run `playwright install chromium`
**Empty Text**: Check if site needs JavaScript
**LLM Fails**: Verify Ollama is running
**Memory Issues**: Reduce concurrency/buffer size

### Debug Commands
```bash
# Enable debug logging
scrapy crawl static_spider -L DEBUG

# Check entity grouping
grep entity_id data/exports/entities_aggregated.csv | sort | uniq -c

# Verify Hamburg detection
grep has_hamburg_reference data/exports/csv/*.csv | grep True | wc -l
```

---

## Recent Updates (2024-01)

### Bug Fixes
- Fixed JSON serialization issue with sets in entity summaries
- Corrected entity relevance tracking initialization
- Improved pandas import handling

### Enhancements
- 6-strategy Hamburg detection
- 14 specific ecosystem roles
- Entity-organized storage
- Dual link saving system
- Comprehensive filtering logic

---

*Last Updated: 2024-01-16*

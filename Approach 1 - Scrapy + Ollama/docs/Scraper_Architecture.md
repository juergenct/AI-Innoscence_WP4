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
- **Hamburg-focused**: Enhanced 6-strategy detection with strong gating (non-Hamburg entities stop immediately)
- **CE detection**: LLM-first CE detection during crawl (keyword fallback)
- **LLM classification (real-time)**: 14 ecosystem roles via local Ollama models; batched during crawl
- **Comprehensive crawl**: Hamburg+CE entities crawled without depth limit for internal links
- **Iterative discovery**: Multi-round crawling with intelligent link filtering (LLM-preferred relevance)
- **Entity-organized storage**: Hierarchical data organization by entity

### Technology Stack
- **Core**: Python 3.8+, Scrapy 2.x
- **JavaScript rendering**: Playwright via scrapy-playwright
- **Text extraction**: trafilatura, BeautifulSoup4
- **Data processing**: pandas, pyarrow
- **LLM**: Ollama with Qwen/qwen3:4b-2507
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

6. **LLM Classification Phase (Real-time, Batched)**
   - LLM pipeline buffers entity contexts and classifies in batches (size 100, flush every 30s or on close)
   - Produces `hamburg_llm`, `ce_llm`, `role_llm`, `classification_confidence`
   - Results gate crawling (mark entities for comprehensive crawl) and influence link filtering
   - Keywords retained as fallback when LLM is unavailable

7. **Data Storage Phase** (Entity-organized)
   - Raw HTML with metadata JSON
   - Processed text with headers
   - Entity summary aggregation
   - Buffered writes to CSV/Parquet/JSON

8. **Link Discovery Phase**
   - Extract all internal and external links
   - Track which entity discovered each link
   - Identify potential new entities
   - Save ALL links for reference

9. **Filtering Phase** (Hamburg+CE)
   - Filter links to only those from relevant entities
   - Create iteration seeds CSV for next round
   - Generate statistics report

10. **Optional Offline LLM Merge**
    - Post-run consolidation step (optional): merge LLM classification into aggregated CSV via `run_scraper.py --classify`

11. **Iteration Loop**
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
- `has_circular_economy_terms`: CE relevance (keyword fallback)
- `has_hamburg_reference`: Hamburg relevance (spider detection)
- `keywords`: Found CE keywords
- `hamburg_llm`: LLM Hamburg boolean (if available)
- `ce_llm`: LLM CE boolean (if available)
- `role_llm`: One of 14 roles (if available)
- `classification_confidence`: 0–1 (if available)

---

## Spiders

### BaseCircularEconomySpider
Abstract base class providing common functionality.

**Key Features:**
- URL loading from CSV or command line
- Entity resolution and relevance tracking
- Hamburg detection (enhanced 6-strategy approach) with strong gating
- Depth-limited crawling (with comprehensive override for Hamburg+CE)
- Statistics collection

**Global Skip Rules (noise reduction):**
- Skip subdomains: `intranet.`, `extranet.`, `collaborating.`, `cloud.`, `login.`, `account.`, `nextcloud.`, `studip.`, `katalog.`, `events.`
- Skip paths (unless depth==0 for initial classification): `/login`, `/wp-admin`, `/account`, `/apps/forms`, `/impressum`, `/datenschutz`

**Entity Relevance Gating:**
- First page of a new entity runs Hamburg detection
- If non-Hamburg: stop crawling this entity immediately; do not follow links
- If Hamburg: continue; if LLM determines CE=true, mark entity for comprehensive crawl
- Comprehensive crawl: internal links are followed without depth limit; external links still respect depth limits
- Candidate pages (contact/about) may be used at depth 0 to help classify

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

### 3. LLMClassificationPipeline (Priority: 250)
- Batched real-time classification (default batch size: 100; flush every 30s)
- Outputs: `hamburg_llm`, `ce_llm`, `role_llm`, `classification_confidence`
- Updates spider gates and comprehensive crawling set
- Caches results to avoid re-classification

### 4. DataStoragePipeline (Priority: 300)
- Entity-organized storage
- Buffered writes
- Multiple formats (CSV, Parquet, JSON)
- Entity summary aggregation

### 5. LinkExtractionPipeline (Priority: 400)
- Extract all links
- Track source entity
- Dual output: ALL links + Filtered seeds (LLM-preferred Hamburg+CE)
- Statistics generation
 - Domain/path skip rules aligned with spider to avoid portals/login/forms in iteration seeds

---

## Utilities

### EntityResolver
Groups pages into logical entities.
- Default: Group by domain
- Academic: Group by first path segment
- Multi-entity domain support

### Name-to-Domain Resolver
Resolves company names to official websites to create high-quality seed lists.
- Input: CSV with `name` (required if `website` missing) and optional `website`/`url` column
- If `website` is provided, websearch is skipped; URL is validated and used
- Otherwise, performs websearch + verification (Impressum/Kontakt, Hamburg signals, .de TLD)
- Outputs:
  - `data/iter_seeds/seeds_from_names.csv` (website)
  - `data/iter_seeds/name_to_domain_mapping.csv` (name→domain mapping with confidence)

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
3. Filter Hamburg+CE relevant (LLM-preferred when available)
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

### Real-time Batched Classification
1. During crawl, the LLM pipeline collects 1–3 sample pages per entity
2. Entities are queued once they meet the minimum samples (default 1)
3. A batch is flushed when 100 entities are queued, 30 seconds have elapsed, or the spider closes
4. Classification runs in a small thread pool; results are cached and applied to subsequent items
5. Results gate crawling (mark CE entities for comprehensive crawl) and are used for iteration seed filtering (preferred over keyword flags)

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
# LLM batching
LLM_CLASSIFY_BATCH_SIZE = 100
LLM_MIN_SAMPLES_PER_ENTITY = 1
LLM_CLASSIFY_FLUSH_INTERVAL_SECS = 30
```

### Setup
```bash
pip install -r requirements.txt
playwright install chromium
ollama pull qwen3:4b
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

# Cleanup legacy/unused data (keep last 2 sessions and generations)
python -m circular_scraper.utils.cleanup_manager --data-dir data --keep-sessions 2 --keep-generations 2
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

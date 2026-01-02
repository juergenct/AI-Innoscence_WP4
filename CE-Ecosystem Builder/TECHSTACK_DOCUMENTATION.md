# Hamburg CE Ecosystem - Tech Stack Documentation

**Version**: 2.0
**Last Updated**: November 2025
**Python Version**: 3.10+

---

## Table of Contents

1. [Overview](#overview)
2. [Core Technologies](#core-technologies)
3. [Technology Architecture](#technology-architecture)
4. [Integration Points](#integration-points)
5. [Installation Guide](#installation-guide)
6. [Version Requirements](#version-requirements)
7. [Technology Roles](#technology-roles)
8. [Performance Characteristics](#performance-characteristics)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Hamburg CE Ecosystem scraper is built on a modern Python stack optimized for **LLM-driven web scraping**, **structured data extraction**, and **large-scale relationship analysis**. The architecture prioritizes:

- ✅ **Local inference** (Ollama) for data privacy and cost control
- ✅ **Async processing** (AsyncIO) for high throughput
- ✅ **Structured extraction** (Instructor + Pydantic) for reliability
- ✅ **Intelligent scraping** (ScrapegraphAI) for complex websites
- ✅ **Robust error handling** with automatic retries and fallbacks

---

## Core Technologies

### 1. Ollama - Local LLM Inference Server

**Role**: Hosts and serves the qwen2.5:32b model for all LLM operations

**Version**: Latest (0.3.0+)
**Website**: https://ollama.ai
**License**: MIT

**Key Features**:
- Local inference (no API costs, data privacy)
- OpenAI-compatible API (`/v1/chat/completions`)
- JSON schema enforcement via GBNF grammar
- Model quantization support (Q4_K_M for memory efficiency)
- Concurrent request handling

**Usage in Pipeline**:
- Stage 1: Hamburg/CE verification
- Stage 2: Entity information extraction (6 parallel calls)
- Stage 4: Relationship analysis and clustering

**Configuration**:
```yaml
llm:
  base_url: 'http://localhost:11434'
  model: 'ollama/qwen2.5:32b-instruct-q4_K_M'
```

**Resource Requirements**:
- VRAM: ~20GB for single model instance
- RAM: ~4GB
- Disk: ~20GB for model storage
- For 4 concurrent entities: 24-28GB total VRAM recommended

---

### 2. Qwen2.5:32B-Instruct (Q4_K_M) - Language Model

**Role**: Core reasoning engine for all extraction, classification, and analysis tasks

**Parameters**: 32 billion (Q4_K_M quantization)
**Context Window**: 32,768 tokens
**Developer**: Alibaba Cloud (Tongyi Qianwen)
**License**: Apache 2.0

**Why Qwen2.5:32B?**
- Superior instruction following compared to 7B/14B models
- Strong multilingual support (English, German)
- Excellent structured output generation
- Efficient Q4_K_M quantization (20GB vs 64GB unquantized)
- Better entity name inference and context understanding

**Model Installation**:
```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

**Temperature Settings**:
- Verification: 0.0 (deterministic classification)
- Extraction: 0.1 (slight flexibility for entity names)
- Relationships: 0.1 (balanced precision/creativity)

---

### 3. ScrapegraphAI - LLM-Guided Web Scraping

**Role**: Intelligent web scraping that navigates websites and extracts relevant information

**Version**: 1.0.0+
**GitHub**: https://github.com/VinciGit00/Scrapegraph-ai
**License**: MIT

**Key Features**:
- LLM-guided navigation (follows links intelligently)
- Multi-page extraction (visits contact, about, impressum pages)
- Dynamic content handling
- Natural language prompts (no CSS selectors needed)
- Integration with Ollama

**Usage in Pipeline**:
- Stage 1: Extract location and CE keywords for verification
- Stage 2: Extract comprehensive entity information from websites

**Integration**:
```python
from scrapegraphai.graphs import SmartScraperGraph

graph = SmartScraperGraph(
    prompt="Extract circular economy information...",
    source=url,
    config=config  # Points to Ollama
)
result = graph.run()
```

**Known Issues & Workarounds**:
- Sometimes raises "Invalid json output" error but includes data in error message
- **Solution**: Error recovery logic extracts content from error message
- **Fallback**: Playwright direct HTML fetch if extraction fails entirely

---

### 4. Instructor - Structured Extraction Library

**Role**: Ensures LLM outputs match Pydantic schemas with automatic validation and retries

**Version**: 1.0.0+
**GitHub**: https://github.com/jxnl/instructor
**License**: MIT

**Key Features**:
- Pydantic v2 integration
- Automatic retry with refined prompts
- OpenAI client patching (works with Ollama)
- JSON mode support (uses Ollama's GBNF grammar)
- Nested model validation

**Usage in Pipeline**:
- **Exclusive to Stage 2**: All entity extraction uses Instructor
- 6 parallel focused extraction calls per entity
- Each call has max_retries=2 for reliability

**Integration**:
```python
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(
    AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Dummy key for Ollama
    ),
    mode=instructor.Mode.JSON
)

result = await client.chat.completions.create(
    model="qwen2.5:32b-instruct-q4_K_M",
    response_model=BasicInfo,  # Pydantic model
    max_retries=2,
    messages=[...]
)
```

**Why Instructor?**
- Alternative considered: **Outlines** (constrained generation)
- **Decision**: Stick with Instructor because:
  - Outlines requires direct model access (transformers/llama.cpp)
  - With Ollama HTTP API, both use same GBNF mechanism
  - Instructor already 81.8% complete in project
  - No performance benefit from switching

---

### 5. Pydantic v2 - Data Validation

**Role**: Defines schemas and validates all extracted data

**Version**: 2.0+
**Website**: https://docs.pydantic.dev/
**License**: MIT

**Key Features**:
- Type-safe data models
- Automatic validation
- Nested model support
- JSON schema generation
- Field validators and constraints

**Usage in Pipeline**:
- All entity profiles validated with Pydantic models
- Nested models for structured fields:
  - `list[CECapability]` instead of `list[dict]`
  - `list[CEActivity]`
  - `list[CENeed]`
  - `list[MentionedPartner]`
  - `list[DiscoveredEntity]`

**Key Models**:
```python
# hamburg_ce_ecosystem/models/entity.py
class EntityProfile(BaseModel):
    url: str
    entity_name: str
    ecosystem_role: EcosystemRole
    ce_capabilities_offered: list[CECapability]
    ce_activities_structured: list[CEActivity]
    ce_needs_requirements: list[CENeed]
    mentioned_partners: list[MentionedPartner]
    discovered_entities: list[DiscoveredEntity]
    # ... 20+ more fields

# hamburg_ce_ecosystem/models/extraction_models.py
class BasicInfo(BaseModel):
    entity_name: str
    ecosystem_role: str
    brief_description: str
    address: str
```

**Pydantic v2 Migration Notes**:
- Use `.model_dump()` instead of `.dict()`
- Use `model_validate()` instead of `parse_obj()`
- Field default factories require `default_factory=list`

---

### 6. Playwright - Browser Automation

**Role**: Fetches raw HTML for preservation and fallback scraping

**Version**: 1.40.0+
**Website**: https://playwright.dev/python/
**License**: Apache 2.0

**Key Features**:
- Headless browser automation
- JavaScript rendering
- Multi-page support
- Network interception
- Screenshot capture (unused in current version)

**Usage in Pipeline**:
- Stage 2: Fetch and save original HTML for each entity
- Fallback: When ScrapegraphAI fails, use Playwright direct fetch

**Integration**:
```python
from hamburg_ce_ecosystem.utils.web_fetcher import fetch_website_content

html_content = fetch_website_content(
    url,
    max_pages=3,  # Follow up to 3 internal links
    timeout=30000  # 30 second timeout
)
```

**Async/Sync Handling**:
- Playwright is synchronous library
- Wrapped with `asyncio.to_thread()` to avoid blocking async loop
- Warning messages expected: "sync API in async context" (harmless)

**Browser Setup**:
```bash
playwright install chromium
```

---

### 7. AsyncIO - Asynchronous Concurrency

**Role**: Enables parallel processing of multiple entities and extraction calls

**Version**: Python 3.10+ standard library
**Documentation**: https://docs.python.org/3/library/asyncio.html

**Key Features**:
- Non-blocking I/O for HTTP and LLM calls
- Semaphore-based concurrency limiting
- `asyncio.gather()` for parallel task execution
- Exception handling with `return_exceptions=True`

**Usage in Pipeline**:
- **Entity-level parallelism**: 4 entities processed concurrently
- **Extraction-level parallelism**: 6 extraction calls per entity in parallel
- **Total parallelism**: 4 entities × 6 calls = 24 concurrent LLM requests

**Key Pattern** (`batch_processor.py`):
```python
async def process_entities_async(entities, max_concurrent=4):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(entity):
        async with semaphore:
            return await process_single_entity(entity)

    tasks = [process_with_semaphore(e) for e in entities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Performance Impact**:
- Without AsyncIO: ~38 entities/hour (sequential)
- With AsyncIO (4 concurrent): ~154 entities/hour (4x speedup)

---

### 8. SQLite - Database

**Role**: Persistent storage for all entities, relationships, and processing state

**Version**: Python 3.10+ standard library (sqlite3)
**Database File**: `data/final/ecosystem.db`

**Schema**:
```sql
-- Verification results (Stage 1)
CREATE TABLE verification_results (
    url TEXT PRIMARY KEY,
    is_hamburg_based BOOLEAN,
    hamburg_confidence REAL,
    is_ce_related BOOLEAN,
    ce_confidence REAL,
    verification_reasoning TEXT,
    should_extract BOOLEAN
);

-- Entity profiles (Stage 2)
CREATE TABLE entity_profiles (
    url TEXT PRIMARY KEY,
    entity_name TEXT,
    ecosystem_role TEXT,
    ce_capabilities_offered JSON,  -- Stored as JSON
    ce_activities_structured JSON,
    -- ... 20+ more fields
);

-- Relationships (Stage 4)
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    entity1_url TEXT,
    entity2_url TEXT,
    relationship_type TEXT,
    confidence REAL,
    source TEXT
);

-- Discovery tracking (Stage 2.5)
CREATE TABLE discovered_entities (
    url TEXT PRIMARY KEY,
    discovered_by_url TEXT,
    discovery_depth INTEGER,
    first_seen_timestamp TEXT
);
```

**JSON Storage**:
- Lists of Pydantic models serialized to JSON for SQLite storage
- Deserialized back to Pydantic models on read

---

### 9. File-Based Caching

**Role**: Cache LLM responses to avoid redundant API calls

**Implementation**: `hamburg_ce_ecosystem/utils/cache.py`
**Cache Location**: `data/.cache/`

**Directory Structure**:
```
data/.cache/
├── verification/        # Stage 1 verification results
│   └── {url_hash}.json
├── extraction/          # Stage 2 extraction results
│   └── {url_hash}.json
└── relationships/       # Stage 4 relationship analysis
    └── {batch_hash}.json
```

**Cache Key Generation**:
```python
import hashlib

def cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()
```

**Cache Behavior**:
- Hit: Return cached result immediately (no LLM call)
- Miss: Call LLM, cache result, return
- Expiration: Manual (delete cache directory to refresh)

**Benefits**:
- Speeds up re-runs by 100x (no LLM calls)
- Enables iterative development
- Resume capability after crashes

---

### 10. Nominatim - Geocoding Service

**Role**: Convert addresses to latitude/longitude coordinates

**Version**: Free service (no API key required)
**Website**: https://nominatim.openstreetmap.org/
**Rate Limit**: 1 request/second

**Usage in Pipeline**:
- Stage 3: Geocode all entity addresses
- Fallback: If address fails, try entity name

**Integration**:
```python
from geopy.geocoders import Nominatim

geocoder = Nominatim(user_agent="hamburg_ce_ecosystem")
location = geocoder.geocode("Große Elbstraße 277, 22767 Hamburg")
# Returns: latitude=53.5481, longitude=9.9368
```

**Caching**:
- All geocoding results cached to CSV
- File: `data/geocode_cache.csv`

---

## Technology Architecture

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT DATA                            │
│                  (CSV with URLs)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 1: VERIFICATION                       │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ScrapegraphAI │─────────│   Ollama     │                  │
│  │  (Extract)   │  Prompt │ (qwen2.5:32b)│                  │
│  └──────────────┘         └──────────────┘                  │
│         │                         │                          │
│         │        ┌────────────────┘                          │
│         ▼        ▼                                           │
│  ┌─────────────────────┐                                    │
│  │  Pydantic Validator │                                    │
│  │ (VerificationResult)│                                    │
│  └─────────────────────┘                                    │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────┐                                    │
│  │  FileCache + SQLite │                                    │
│  └─────────────────────┘                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ (should_extract=True)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2: EXTRACTION                         │
│  ┌──────────────┐                                            │
│  │ScrapegraphAI │────┐                                       │
│  │  (Extract)   │    │                                       │
│  └──────────────┘    │                                       │
│  ┌──────────────┐    │                                       │
│  │  Playwright  │────┤ Raw Content                           │
│  │ (HTML Fetch) │    │                                       │
│  └──────────────┘    │                                       │
│         │             │                                       │
│         ▼             ▼                                       │
│  ┌──────────────────────────────────────────┐               │
│  │     INSTRUCTOR PARALLEL EXTRACTION        │               │
│  │  ┌────────────┐  ┌────────────┐          │               │
│  │  │ BasicInfo  │  │ContactInfo │          │               │
│  │  │ Extraction │  │ Extraction │          │               │
│  │  └─────┬──────┘  └─────┬──────┘          │               │
│  │        │                │                 │               │
│  │  ┌─────┴────────────────┴──────┐         │               │
│  │  │  AsyncIO Parallel (6 calls)  │        │               │
│  │  │  ┌───────────────┐            │        │               │
│  │  │  │    Ollama     │            │        │               │
│  │  │  │(qwen2.5:32b)  │            │        │               │
│  │  │  └───────────────┘            │        │               │
│  │  └──────────────────────────────┘         │               │
│  │  ┌────────────┐  ┌────────────┐          │               │
│  │  │CECapability│  │CEActivities│          │               │
│  │  │ Extraction │  │ Extraction │          │               │
│  │  └─────┬──────┘  └─────┬──────┘          │               │
│  │        │                │                 │               │
│  │  ┌─────┴────────────────┴──────┐         │               │
│  │  │   Pydantic Validation        │        │               │
│  │  │  (Nested Models)              │        │               │
│  │  └──────────────────────────────┘         │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│                     ▼                                         │
│  ┌────────────────────────────────┐                         │
│  │   EntityProfile (Merged)       │                         │
│  └────────────────────────────────┘                         │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────┐                                    │
│  │  FileCache + SQLite │                                    │
│  │  + Raw HTML Storage │                                    │
│  └─────────────────────┘                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 3: GEOCODING                          │
│  ┌──────────────┐                                            │
│  │  Nominatim   │─────► (lat, lon)                          │
│  │  Geocoder    │                                            │
│  └──────────────┘                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 4: RELATIONSHIP ANALYSIS                  │
│  ┌──────────────────────────────────────────┐               │
│  │         Partner Matching (LLM)            │               │
│  │  ┌────────────────────────────┐           │               │
│  │  │  Ollama (qwen2.5:32b)      │           │               │
│  │  │  Semantic matching          │           │               │
│  │  └────────────────────────────┘           │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│  ┌─────────────────┴────────────────────────┐               │
│  │  Clustering (Capabilities/Needs/Activity) │               │
│  │  ┌────────────────────────────┐           │               │
│  │  │  Ollama (qwen2.5:32b)      │           │               │
│  │  │  LLM-based grouping         │           │               │
│  │  └────────────────────────────┘           │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│  ┌─────────────────┴────────────────────────┐               │
│  │  Synergy & Gap Analysis                   │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────┐                                    │
│  │  Relationships JSON │                                    │
│  │  Clusters JSON      │                                    │
│  │  Insights JSON      │                                    │
│  └─────────────────────┘                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 5: BUILD GRAPH                         │
│  ┌──────────────────────────────────────────┐               │
│  │  NetworkX Graph Construction              │               │
│  │  ├─ Nodes (Entities)                      │               │
│  │  ├─ Edges (Relationships)                 │               │
│  │  └─ Attributes (Capabilities, etc.)       │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────┐                                    │
│  │  ecosystem_map.json │                                    │
│  │  ecosystem.db       │                                    │
│  │  CSV exports        │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Ollama ↔ ScrapegraphAI

**Connection**: ScrapegraphAI uses Ollama via HTTP API

**Configuration** (`scrape_config.yaml`):
```yaml
llm:
  model: 'ollama/qwen2.5:32b-instruct-q4_K_M'
  base_url: 'http://localhost:11434'
```

**Data Flow**:
1. ScrapegraphAI sends prompt + HTML content to Ollama
2. Ollama generates extraction instructions
3. ScrapegraphAI parses HTML based on LLM guidance
4. Returns structured data (or error message with content)

---

### 2. Ollama ↔ Instructor ↔ OpenAI Client

**Connection**: Instructor patches OpenAI client to add Pydantic validation

**Setup**:
```python
from openai import AsyncOpenAI
import instructor

client = instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)
```

**Data Flow**:
1. Instructor converts Pydantic model to JSON schema
2. Sends schema + prompt to Ollama via OpenAI-compatible API
3. Ollama uses GBNF grammar to enforce JSON structure
4. Instructor validates response against Pydantic model
5. On failure: Refines prompt and retries (up to max_retries)

**Key Insight**: Instructor + Ollama = Guaranteed structured output

---

### 3. AsyncIO ↔ All Components

**Connection**: AsyncIO orchestrates all I/O operations

**Components Using AsyncIO**:
- Instructor extraction calls (AsyncOpenAI client)
- ScrapegraphAI (async HTTP requests)
- Database operations (SQLite via asyncio wrappers)
- File I/O (async file operations)

**Exception**: Playwright is synchronous, wrapped with `asyncio.to_thread()`

---

### 4. Pydantic ↔ SQLite

**Connection**: Pydantic models serialized to JSON for SQLite storage

**Serialization**:
```python
# Save to database
profile_dict = profile.model_dump()
profile_dict['ce_capabilities_offered'] = json.dumps(
    [cap.model_dump() for cap in profile.ce_capabilities_offered]
)
cursor.execute("INSERT INTO entity_profiles VALUES (?)", profile_dict)

# Load from database
row = cursor.fetchone()
profile = EntityProfile(
    **row,
    ce_capabilities_offered=[
        CECapability(**cap) for cap in json.loads(row['ce_capabilities_offered'])
    ]
)
```

---

## Installation Guide

### 1. System Requirements

- **OS**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.10 or higher
- **RAM**: 32GB minimum, 64GB recommended
- **VRAM**: 24-28GB for 4 concurrent entities
- **Disk**: 30GB free space (model + data)

### 2. Install Ollama

**Linux/macOS**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows**: Download from https://ollama.ai/download

**Verify Installation**:
```bash
ollama --version
```

### 3. Pull Model

```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

**Verify Model**:
```bash
ollama list
# Should show: qwen2.5:32b-instruct-q4_K_M
```

### 4. Install Python Dependencies

**Clone Repository**:
```bash
cd ~/Documents/AI-Innoscence_Ecosystem
cd "CE-Ecosystem Builder"
```

**Create Virtual Environment**:
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows
```

**Install Requirements**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install Playwright Browser

```bash
playwright install chromium
```

### 6. Configuration

**Create Config** (if not exists):
```bash
cp hamburg_ce_ecosystem/config/scrape_config.yaml.example \
   hamburg_ce_ecosystem/config/scrape_config.yaml
```

**Edit Config**:
```yaml
llm:
  model: 'ollama/qwen2.5:32b-instruct-q4_K_M'
  base_url: 'http://localhost:11434'
  temperature: 0.0

scraper:
  max_concurrent_entities: 4  # Adjust based on VRAM
```

### 7. Start Ollama Server

```bash
ollama serve
# Should output: Listening on http://127.0.0.1:11434
```

### 8. Test Installation

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test Python imports
python -c "import instructor, scrapegraphai, pydantic; print('OK')"

# Test pipeline
python -m hamburg_ce_ecosystem.scrapers.batch_processor --help
```

---

## Version Requirements

### Core Dependencies

```txt
# LLM & AI
ollama>=0.3.0
instructor>=1.0.0
openai>=1.0.0  # For Instructor integration

# Web Scraping
scrapegraphai>=1.0.0
playwright>=1.40.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Data Validation
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Async & HTTP
httpx>=0.24.0
aiofiles>=23.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Geocoding
geopy>=2.3.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
tqdm>=4.65.0
colorlog>=6.7.0
```

### Python Version Compatibility

- **Python 3.10**: ✅ Fully supported (recommended)
- **Python 3.11**: ✅ Fully supported
- **Python 3.12**: ⚠️ May have Playwright compatibility issues
- **Python 3.9**: ❌ Not supported (AsyncIO features required)

---

## Technology Roles

### Data Flow by Stage

| Stage | Primary Tech | Supporting Tech | Output |
|-------|--------------|----------------|--------|
| 1. Verification | ScrapegraphAI | Ollama, Pydantic | `verification_results.csv` |
| 2. Extraction | Instructor | Ollama, Playwright, Pydantic | `entity_profiles.json` |
| 3. Geocoding | Nominatim | geopy | Updated profiles with coordinates |
| 4. Relationships | Ollama | Pydantic | `relationships.json`, `clusters.json` |
| 5. Graph Build | SQLite | pandas | `ecosystem.db`, `ecosystem_map.json` |

### Technology by Function

| Function | Technologies |
|----------|--------------|
| **LLM Inference** | Ollama, Qwen2.5:32B |
| **Structured Extraction** | Instructor, Pydantic |
| **Web Scraping** | ScrapegraphAI, Playwright, BeautifulSoup |
| **Concurrency** | AsyncIO, Semaphore |
| **Data Storage** | SQLite, JSON, CSV, FileCache |
| **Geocoding** | Nominatim, geopy |
| **Error Handling** | Pydantic validation, try/except, retries |
| **Configuration** | PyYAML, python-dotenv |
| **Logging** | Python logging, colorlog |

---

## Performance Characteristics

### Processing Speed

| Metric | Value |
|--------|-------|
| Single entity (verification) | ~15 seconds |
| Single entity (extraction) | ~93 seconds |
| Throughput (4 concurrent) | ~154 entities/hour |
| Total for 2,500 entities | ~16-18 hours |

### Resource Usage

| Resource | Usage |
|----------|-------|
| VRAM (single entity) | ~20GB |
| VRAM (4 concurrent) | ~24-28GB |
| RAM | ~8-12GB |
| CPU (during inference) | 60-80% (depends on model) |
| Disk (cache) | ~500MB per 1000 entities |
| Network (Ollama) | Local (no external calls) |

### Bottlenecks

1. **LLM Inference Time**: 93 seconds per entity (dominant)
   - **Mitigation**: AsyncIO parallelism (4 concurrent)

2. **VRAM Limits**: Cannot exceed concurrent capacity
   - **Mitigation**: Adjust `max_concurrent_entities` based on hardware

3. **Playwright Sync API**: Slight overhead from async wrapper
   - **Impact**: Negligible (<1 second per entity)

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Refused

**Symptoms**:
```
ConnectionError: Cannot connect to Ollama at http://localhost:11434
```

**Solutions**:
- Start Ollama server: `ollama serve`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Verify firewall allows port 11434

---

#### 2. Model Not Found

**Symptoms**:
```
Error: model 'qwen2.5:32b-instruct-q4_K_M' not found
```

**Solutions**:
- Pull model: `ollama pull qwen2.5:32b-instruct-q4_K_M`
- List models: `ollama list`
- Check model name in config matches exactly

---

#### 3. Out of VRAM

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `max_concurrent_entities` in config (try 2 or 1)
- Use smaller model (e.g., `qwen2.5:14b` - update WORKFLOW_DOCUMENTATION.md)
- Close other GPU applications
- Monitor VRAM: `nvidia-smi -l 1`

---

#### 4. Playwright Not Installed

**Symptoms**:
```
playwright._impl._api_types.Error: Executable doesn't exist
```

**Solutions**:
- Install browsers: `playwright install chromium`
- Verify installation: `playwright --version`

---

#### 5. Pydantic Validation Errors

**Symptoms**:
```
ValidationError: 1 validation error for EntityProfile
  ce_capabilities_offered.0.capability_name
    field required (type=value_error.missing)
```

**Solutions**:
- Check LLM is using correct model (32B not 7B)
- Verify temperature settings (0.1 for extraction)
- Review prompt in `config/extraction_prompts.py`
- Check Instructor retries (should be max_retries=2)

---

#### 6. ScrapegraphAI "Invalid json output" Errors

**Symptoms**:
```
Exception: Invalid json output: {"entity_name": "Example"}
For troubleshooting, visit: ...
```

**Solutions**:
- **Expected behavior**: Error recovery extracts content automatically
- Check logs for "Recovered content from ScrapegraphAI error"
- If recovery fails, entity uses Playwright fallback
- Silent recovery is intentional (logged at DEBUG level)

---

#### 7. Async Sync API Warnings

**Symptoms**:
```
Warning: sync API called from async context
```

**Solutions**:
- **Expected behavior**: Playwright wrapped with `asyncio.to_thread()`
- Warnings are harmless, entities still process successfully
- Can suppress by setting log level to ERROR in config

---

### Logging Configuration

**Enable DEBUG logging** (`hamburg_ce_ecosystem/utils/logging_setup.py`):
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Log Files**:
- Errors: `logs/scraping_errors.log`
- Relationships: `logs/relationship_analysis.log`
- General: Console output

---

## References & Resources

### Official Documentation

- **Ollama**: https://ollama.ai/docs
- **Qwen2.5**: https://github.com/QwenLM/Qwen2.5
- **ScrapegraphAI**: https://scrapegraph-ai.readthedocs.io/
- **Instructor**: https://python.useinstructor.com/
- **Pydantic**: https://docs.pydantic.dev/
- **Playwright**: https://playwright.dev/python/
- **AsyncIO**: https://docs.python.org/3/library/asyncio.html

### Community & Support

- **Ollama GitHub**: https://github.com/ollama/ollama/issues
- **ScrapegraphAI GitHub**: https://github.com/VinciGit00/Scrapegraph-ai/issues
- **Instructor GitHub**: https://github.com/jxnl/instructor/issues
- **Pydantic GitHub**: https://github.com/pydantic/pydantic/issues

### Related Research

- **GBNF Grammar (Ollama)**: Constrained generation using Backus-Naur Form
- **Instructor vs Outlines**: Comparison of structured extraction approaches
- **Qwen2.5 Technical Report**: Model architecture and capabilities

---

**End of Documentation**

# Hamburg CE Ecosystem Scraper - Workflow Documentation

**Version**: 2.0 (O(n) Architecture)
**Last Updated**: October 29, 2025
**Complexity**: O(n) - Linear scaling

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Output Files](#output-files)
7. [Key Features](#key-features)

---

## Overview

This scraper identifies and analyzes circular economy (CE) actors in Hamburg, Germany. It uses a **two-stage LLM approach** with **O(n) complexity** for relationship analysis.

**Key Capabilities**:
- ✅ Verifies Hamburg location and CE relevance
- ✅ Extracts CE-specific capabilities, needs, and activities
- ✅ Saves both original HTML and extracted content
- ✅ Discovers new entities iteratively (up to 3 levels deep)
- ✅ Matches partners using LLM semantic understanding
- ✅ Clusters capabilities, needs, and activities
- ✅ Identifies synergies and ecosystem gaps
- ✅ Performs geocoding for spatial analysis

**Performance**: Processes 2,500+ entities in hours (vs. days with old O(n²) approach)

---

## Architecture

### Two-Stage LLM Approach

**Stage 1: ScrapegraphAI** (Flexible Extraction)
- Visits multiple pages on each website
- Extracts information in any format (JSON, text, etc.)
- Handles messy/incomplete data gracefully

**Stage 2: Ollama + Pydantic** (Structured Validation)
- Takes Stage 1 output and structures it
- Guarantees valid JSON with Pydantic models
- No schema errors - guaranteed structure

### Model Strategy

| Stage | Model | Purpose | VRAM |
|-------|-------|---------|------|
| Verification | `qwen2.5:32b-instruct-q4_K_M` | Hamburg/CE classification | ~20GB |
| Extraction | `qwen2.5:32b-instruct-q4_K_M` | Quality extraction with Instructor library | ~20GB |
| Matching/Clustering | `qwen2.5:32b-instruct-q4_K_M` | Semantic understanding for relationships | ~20GB |

**Note**: All stages use the same 32B parameter model (Q4_K_M quantization) to ensure consistent quality throughout the pipeline. This model provides superior extraction accuracy compared to smaller models like 7B/14B variants.

---

## Pipeline Stages

### Stage 1: Verification

**Purpose**: Filter entities by Hamburg location + CE relevance

**Input**: List of URLs (CSV, JSON, or database)

**Process**:
1. ScrapegraphAI extracts location and CE keywords from website
2. Ollama + Pydantic validates and structures results
3. LLM determines: `is_hamburg_based`, `is_ce_related`, `should_extract`

**Output**: `data/verified/verification_results.csv`

**Key Logic**:
- Hamburg detection: Postal codes 20000-22999, "+49 40", Hamburg districts
- CE detection: Keywords like Kreislaufwirtschaft, Recycling, Nachhaltigkeit, Zero Waste
- Impressum analysis as fallback

---

### Stage 2: Extraction

**Purpose**: Extract detailed CE profiles

**Input**: Entities where `should_extract=True` from Stage 1

**Process**:
1. ScrapegraphAI extracts comprehensive information from website
2. Fetches original HTML via Playwright for preservation
3. Ollama + Pydantic structures into EntityProfile
4. Saves both raw HTML and extracted text

**Extracted Fields**:

**Basic Info**:
- Entity name, ecosystem role, contact persons, emails, phone numbers
- Brief description, CE relation, address

**CE-Focused Structured Data** (NEW):
- `ce_capabilities_offered`: What entity provides **specifically for CE** (e.g., "Recycling services", "Circular design consulting")
- `ce_needs_requirements`: What entity needs **specifically for CE** (e.g., "Sustainable materials", "CE partnerships")
- `ce_activities_structured`: Detailed CE activities with categories
- `mentioned_partners`: Partners/collaborators mentioned on website with context
- `discovered_entities`: New entities found on website with URLs

**Output**:
- `data/extracted/entity_profiles.json`
- `data/raw_html/{url_hash}/{timestamp}_raw.html` (original HTML)
- `data/raw_html/{url_hash}/{timestamp}_extracted.txt` (ScrapegraphAI output)

**Important**: Only extracts CE-related capabilities/needs. General business capabilities like "marketing" or "IT support" are ignored unless CE-specific (e.g., "CE marketing campaigns").

---

### Instructor Library: Parallel Focused Extraction

**Critical Implementation Detail**: Stage 2 uses the **Instructor library** for reliable, structured extraction with automatic validation and retries.

**Architecture**: Instead of a single monolithic extraction call, the system makes **6 parallel focused extraction calls** for maximum reliability and speed:

| Call | Response Model | Purpose | Temperature | Max Retries |
|------|----------------|---------|-------------|-------------|
| 1. Basic Info | `BasicInfo` | Entity name, role, description, address | 0.1 | 2 |
| 2. Contact Info | `ContactInfo` | Emails, phone numbers, contact persons | 0.0 | 2 |
| 3. CE Capabilities | `CECapabilities` | What entity offers to CE ecosystem | 0.1 | 2 |
| 4. CE Activities | `CEActivities` | CE activities performed, relation to CE | 0.1 | 2 |
| 5. CE Needs | `CENeeds` | CE-related needs and requirements | 0.1 | 2 |
| 6. Partnerships | `Partnerships` | Mentioned partners, discovered entities | 0.0 | 2 |

**Why Instructor?**
- **Pydantic v2 Integration**: Automatic validation using nested Pydantic models (e.g., `list[CECapability]` instead of `list[dict]`)
- **Automatic Retries**: Each call retries up to 2 times on validation failure
- **JSON Mode**: Uses Ollama's JSON schema enforcement (GBNF grammar) for guaranteed structure
- **Focused Prompts**: Each call has a specialized prompt for that specific domain
- **Error Recovery**: Failed calls return safe defaults without breaking the pipeline

**Key Implementation** (`hamburg_ce_ecosystem/utils/instructor_extraction.py`):
```python
class InstructorExtractor:
    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:32b-instruct-q4_K_M"):
        self.client = instructor.from_openai(
            AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,  # Use Ollama JSON mode
        )

    async def extract_all_parallel(self, text: str, url: str) -> dict:
        # Launch all 6 extractions concurrently
        results = await asyncio.gather(
            self.extract_basic_info(text, url),
            self.extract_contacts(text),
            self.extract_ce_capabilities(text, url),
            self.extract_ce_activities(text),
            self.extract_ce_needs(text),
            self.extract_partnerships(text),
            return_exceptions=True,  # Continue even if one fails
        )
        # Merge results into single dictionary
        return merged
```

**Performance**:
- **Total Time per Entity**: ~93 seconds (6 parallel calls + ScrapegraphAI extraction)
- **Concurrency**: 4 entities processed simultaneously (AsyncIO semaphore)
- **Throughput**: ~154 entities/hour

**Nested Pydantic Models** (prevents validation errors):
```python
# OLD (prone to validation errors):
ce_capabilities_offered: list[dict]  # LLM can use wrong field names

# NEW (strict validation):
ce_capabilities_offered: list[CECapability]  # Pydantic enforces schema
```

**Text Truncation**: CE capabilities and activities extraction truncates text to 8000 characters to prevent context overflow while preserving schema instructions.

---

### AsyncIO Concurrency Architecture

**Purpose**: Process multiple entities simultaneously while preventing resource exhaustion.

**Implementation**:
- **Semaphore-based Concurrency**: Limits parallel entity processing to 4 concurrent entities
- **Async/Await Pattern**: All I/O operations (HTTP, LLM calls) are non-blocking
- **Graceful Degradation**: Failed entities don't block the queue

**Key Components**:

1. **Batch Processing with Semaphore** (`hamburg_ce_ecosystem/scrapers/batch_processor.py`):
```python
async def process_entities_async(self, entities: list, max_concurrent: int = 4):
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit to 4 concurrent

    async def process_with_semaphore(entity):
        async with semaphore:
            return await self.process_single_entity(entity)

    tasks = [process_with_semaphore(entity) for entity in entities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

2. **Instructor Parallel Extraction** (per entity):
   - Within each entity, 6 extraction calls run in parallel via `asyncio.gather()`
   - Total parallelism: 4 entities × 6 calls = 24 concurrent LLM requests

3. **Playwright Integration**:
   - Playwright is synchronous, wrapped with `asyncio.to_thread()` to avoid blocking
   - Warning messages about "sync API in async context" are expected and harmless

**Performance Impact**:
- **Single Entity Processing Time**: ~93 seconds
- **4 Entities Concurrently**: ~93 seconds (not 372 seconds!)
- **Throughput**: ~154 entities/hour instead of ~38/hour (4x speedup)

**Configuration**:
```yaml
scraper:
  max_concurrent_entities: 4  # Adjust based on available VRAM/CPU
  timeout: 120  # Seconds per extraction attempt
```

**Resource Management**:
- **VRAM**: 4 concurrent entities × 20GB model ≈ requires 24-28GB total VRAM (with overhead)
- **CPU**: Async I/O minimizes CPU blocking
- **Network**: Ollama API calls are queued by Ollama server internally

---

### Stage 2.5: Discovered Entity Processing (NEW)

**Purpose**: Iteratively discover and process new entities

**Input**: `discovered_entities` from all Stage 2 profiles

**Process**:
1. **Aggregate**: Collect all discovered entities from extraction
2. **Deduplicate**: LLM identifies duplicates semantically (e.g., "HCU" = "HafenCity Universität Hamburg")
3. **Filter**: Remove entities already in database
4. **Track**: Create discovery records (who discovered whom)
5. **Queue**: Add new entities to processing pipeline
6. **Iterate**: Process up to `max_discovery_depth` levels (default: 3)

**Discovery Chain Example**:
```
Hamburg University → Startup A → Partner B → Supplier C
```

**Output**:
- `data/discovered_entities/discovery_records.json`
- New entities added to verification queue

**Configuration**:
```yaml
discovered_entity_processing:
  enable: true
  max_discovery_depth: 3
  batch_size: 30
  min_deduplication_confidence: 0.7
```

---

### Stage 3: Geocoding

**Purpose**: Add latitude/longitude coordinates

**Input**: Entity profiles with addresses

**Process**:
1. Uses Nominatim geocoder
2. Tries full address first
3. Falls back to entity name if address fails
4. Caches results to CSV

**Output**: Updated profiles with `latitude` and `longitude`

---

### Stage 4: Relationship Analysis & Clustering

**Purpose**: Identify partnerships, cluster data, and discover insights

**Input**: All extracted entity profiles

**Process** (All O(n) complexity):

#### 4.1 Partnership Extraction (O(n))
- For each entity's `mentioned_partners`, use LLM to match against database
- Semantic matching: handles abbreviations, variations, context
- Creates relationships with confidence scores
- Marks bidirectional relationships

#### 4.2 CE Capability Clustering (O(n))
- Aggregates all `ce_capabilities_offered` from all entities
- LLM groups into 10-15 meaningful clusters
- Examples: "Recycling Infrastructure", "Circular Design Services", "CE Consulting"

#### 4.3 CE Need Clustering (O(n))
- Aggregates all `ce_needs_requirements` from all entities
- LLM groups into 10-15 meaningful clusters
- Examples: "Sustainable Materials", "Recycling Partnerships", "CE Funding"

#### 4.4 CE Activity Clustering (O(n))
- Aggregates all `ce_activities_structured` from all entities
- LLM groups into 8-12 meaningful clusters
- Examples: "Waste Management", "Repair Services", "CE Research"

#### 4.5 Relationship Type Discovery
- Matches capability clusters to need clusters
- Creates potential synergy relationships
- Types: partnerships, knowledge transfer, supply chains, collaborations

#### 4.6 Synergy Identification
- LLM analyzes ecosystem for collaboration opportunities
- Types: complementary activities, value chains, geographic clusters, shared resources

#### 4.7 Gap Analysis
- LLM identifies missing elements
- Underrepresented roles, broken value chains, geographic gaps
- Provides actionable recommendations

**Output**:
- `data/relationships/relationships.json` (15-25 relationship types)
- `data/relationships/clusters.json` (capability, need, activity clusters)
- `data/relationships/ecosystem_insights.json` (synergies, gaps, recommendations)

**Configuration**:
```yaml
entity_matching:
  batch_size: 50
  min_confidence: 0.6

relationship_analysis:
  clustering:
    enable: true
    capability_clusters: 15
    need_clusters: 15
    activity_clusters: 12
```

---

### Stage 5: Build Ecosystem Graph

**Purpose**: Create final network graph and export data

**Input**: Entities, relationships, clusters, insights

**Process**:
1. Creates nodes (entities) with properties
2. Creates edges (relationships) with types
3. Integrates cluster information
4. Adds ecosystem insights

**Output**:
- `data/final/ecosystem_map.json` (full graph structure)
- `data/final/ecosystem_entities.csv` (entity list)
- `data/final/ecosystem_relationships.csv` (relationship list)
- `data/final/ecosystem.db` (SQLite database)

---

## Configuration

### Main Config File: `config/scrape_config.yaml`

```yaml
# LLM Configuration
llm:
  model: 'ollama/qwen2.5:32b-instruct-q4_K_M'
  base_url: 'http://localhost:11434'
  temperature: 0.0

# Stage-Specific Overrides (optional)
verification:
  temperature: 0.0  # Deterministic classification

extraction:
  temperature: 0.1  # Slightly flexible for entity name inference

# Discovered Entity Processing (Stage 2.5)
discovered_entity_processing:
  enable: true
  max_discovery_depth: 3
  batch_size: 30
  min_deduplication_confidence: 0.7

# Entity Matching (LLM-based)
entity_matching:
  batch_size: 50
  min_confidence: 0.6

# Relationship Analysis (O(n))
relationship_analysis:
  clustering:
    enable: true
    capability_clusters: 15
    need_clusters: 15
    activity_clusters: 12
```

### Prompts Configuration

**Verification Prompts**: `config/verification_prompts.py`
- `VERIFICATION_PROMPT`: Hamburg location and CE relevance verification
- Uses postal codes, phone prefixes, CE keywords for classification

**Extraction Prompts**: `config/extraction_prompts.py`
- `BASIC_INFO_PROMPT`: Entity identification and basic information
- `CONTACT_INFO_PROMPT`: Email, phone, contact person extraction
- `CE_CAPABILITIES_PROMPT`: CE capabilities offered to ecosystem
- `CE_ACTIVITIES_PROMPT`: CE activities and relation description
- `CE_NEEDS_PROMPT`: CE needs and requirements identification
- `PARTNERSHIPS_PROMPT`: Partner mentions and entity discovery

Each prompt is focused on a specific extraction domain and designed to work with Instructor's automatic validation and retry mechanism.

**Relationship Prompts**: `config/relationship_prompts.py`

Contains prompts for:
- Entity deduplication (semantic matching)
- Partner matching to database entities
- Capability/need/activity clustering
- Synergy detection
- Gap analysis

---

## Running the Pipeline

### Prerequisites

1. **Install Ollama** and pull required model:
   ```bash
   ollama pull qwen2.5:32b-instruct-q4_K_M
   ```
   **Requirements**: ~20GB VRAM, ~24-28GB total for 4 concurrent entities

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare input data**: CSV with `url` column

### Option 1: Run Full Pipeline

```bash
python run_full_pipeline.py \
  --input data/input/entities.csv \
  --config hamburg_ce_ecosystem/config/scrape_config.yaml
```

### Option 2: Run Individual Stages

```bash
# Stage 1: Verification
python -m hamburg_ce_ecosystem.scrapers.batch_processor \
  --stage verification \
  --input data/input/entities.csv

# Stage 2: Extraction
python -m hamburg_ce_ecosystem.scrapers.batch_processor \
  --stage extraction

# Stage 2.5: Discovered Entity Processing
python -m hamburg_ce_ecosystem.scrapers.discovered_entity_processor

# Stage 3: Geocoding
python -m hamburg_ce_ecosystem.scrapers.batch_processor \
  --stage geocoding

# Stage 4: Relationship Analysis
python -m hamburg_ce_ecosystem.scrapers.batch_processor \
  --stage relationships

# Stage 5: Build Graph
python -m hamburg_ce_ecosystem.scrapers.batch_processor \
  --stage build_graph
```

### Monitoring Progress

- **Logs**: `logs/scraping_errors.log`, `logs/relationship_analysis.log`
- **Cache**: `.cache/` directories store LLM responses
- **Database**: `data/final/ecosystem.db` tracks all processed entities

---

## Output Files

### Directory Structure

```
data/
├── input/
│   └── entities.csv                    # Input URLs
├── verified/
│   └── verification_results.csv        # Stage 1 output
├── extracted/
│   └── entity_profiles.json            # Stage 2 output
├── raw_html/
│   └── {url_hash}/
│       ├── {timestamp}_raw.html        # Original HTML (Playwright)
│       ├── {timestamp}_extracted.txt   # ScrapegraphAI output
│       └── {timestamp}_metadata.json   # Metadata
├── discovered_entities/
│   └── discovery_records.json          # Stage 2.5 output
├── relationships/
│   ├── relationships.json              # Stage 4 partnerships
│   ├── clusters.json                   # Stage 4 clusters
│   └── ecosystem_insights.json         # Stage 4 insights
└── final/
    ├── ecosystem_map.json              # Full graph (Stage 5)
    ├── ecosystem_entities.csv          # Entity list
    ├── ecosystem_relationships.csv     # Relationship list
    └── ecosystem.db                    # SQLite database
```

### Database Schema

**Tables**:
- `verification_results`: Hamburg/CE verification results
- `entity_profiles`: Detailed entity information (with CE-focused fields)
- `relationships`: Partnerships and connections (with discovery chains)
- `clusters`: Capability/need/activity clusters
- `discovered_entities`: Entity discovery tracking
- `ecosystem_insights`: Synergies, gaps, recommendations

---

## Key Features

### 1. CE-Focused Extraction

**Only extracts circular economy related data**:
- ✅ "Recycling infrastructure", "Circular design consulting"
- ❌ "Marketing services", "Office space" (unless CE-specific)

### 2. Dual HTML Storage

Saves both:
- **Original HTML**: Complete webpage preservation
- **Extracted text**: ScrapegraphAI processed content

### 3. LLM-Based Semantic Matching

Goes beyond string matching:
- Matches "HCU" to "HafenCity Universität Hamburg"
- Handles "HafenCity University" vs "HafenCity Universität Hamburg"
- Context-aware matching with confidence scores

### 4. Iterative Entity Discovery

Automatically expands the ecosystem:
- Discovers entities mentioned on websites
- Processes them through verification → extraction
- Tracks discovery chains (A → B → C → D)
- Configurable depth limit (default: 3 levels)

### 5. Three-Dimensional Clustering

Provides ecosystem insights:
- **Capability clusters**: What ecosystem can provide
- **Need clusters**: What ecosystem is seeking
- **Activity clusters**: What's actually happening

### 6. O(n) Complexity

**Performance improvement**:
- **Old approach**: O(n²) = 2,500² = 6.25 million comparisons → ~6-7 days
- **New approach**: O(n) = 2,500 entities → ~hours
- **Speedup**: 100-150x faster

### 7. Comprehensive Relationship Discovery

Identifies 15-25 relationship types:
- Direct partnerships (from website mentions)
- Potential synergies (from capability-need matching)
- Knowledge transfer (research → industry)
- Supply chains (material flows)
- Network collaborations (similar activities)
- Discovery relationships (who found whom)

---

## Troubleshooting

### Common Issues

**Issue**: "Ollama connection refused"
- **Solution**: Start Ollama server: `ollama serve`

**Issue**: "Model not found"
- **Solution**: Pull required model: `ollama pull qwen2.5:32b-instruct-q4_K_M`

**Issue**: "ScrapegraphAI timeout"
- **Solution**: Increase timeout in `scrape_config.yaml`: `timeout: 60`

**Issue**: "No CE capabilities extracted"
- **Solution**: Check if prompts are too strict. Review `config/prompts.py`

**Issue**: "Too few relationship types"
- **Solution**: Adjust clustering settings in `scrape_config.yaml`

### Performance Optimization

1. **Adjust concurrency**: Increase `max_concurrent_entities` if you have more VRAM (default: 4)
2. **Enable caching**: All LLM responses cached in `.cache/` directories
3. **AsyncIO parallelism**: 6 extraction calls per entity run in parallel
4. **Adjust discovery depth**: Lower `max_discovery_depth` if too slow (default: 3)
5. **Text truncation**: Already implemented for CE capabilities/activities (8000 chars)
6. **Instructor retries**: Max 2 retries per extraction call prevents infinite loops

### Error Recovery & Resilience

The pipeline includes multiple layers of error recovery to ensure robust processing:

**1. ScrapegraphAI Error Recovery** (`verification_scraper.py:104-127`)
- **Issue**: ScrapegraphAI sometimes fails with "Invalid json output" but includes useful content in error message
- **Solution**: Extract content from error message automatically
- **Fallback**: If extraction fails, use Playwright to fetch raw HTML
- **Logging**: Errors logged at DEBUG level (silent recovery)

```python
try:
    scrapegraph_output = graph.run()
except Exception as e:
    if "Invalid json output:" in str(e):
        # Extract content from error message
        extracted_text = str(e).split("Invalid json output:", 1)[1].strip()
    else:
        # Use Playwright fallback
        extracted_text = fetch_website_content(url)
```

**2. Instructor Automatic Retries** (`instructor_extraction.py:76, 116, 152, etc.`)
- **Issue**: LLM may produce invalid JSON or violate schema
- **Solution**: Instructor automatically retries up to 2 times with refined prompts
- **Max Retries**: 2 per extraction call (configurable)
- **Validation**: Pydantic v2 validates structure after each attempt

**3. Parallel Extraction Exception Handling** (`instructor_extraction.py:310-370`)
- **Issue**: One failed extraction call shouldn't break entire entity
- **Solution**: `asyncio.gather(..., return_exceptions=True)` continues processing
- **Fallback**: Failed calls return safe defaults (empty lists, default values)
- **Example**: If contact extraction fails, returns `ContactInfo(contact_persons=[], emails=[], phone_numbers=[])`

**4. Playwright Async/Sync Warnings** (Expected & Harmless)
- **Issue**: Playwright sync API called in async context generates warnings
- **Solution**: Wrapped with `asyncio.to_thread()` for proper async execution
- **Impact**: Warning messages are expected, entities still process successfully
- **Logging**: Can be suppressed by adjusting log level

**5. Entity-Level Recovery** (`batch_processor.py`)
- **Issue**: One entity failure shouldn't stop entire batch
- **Solution**: Each entity wrapped in try/except, failures logged but don't propagate
- **Database**: Failed entities marked in database for manual review
- **Resume**: Pipeline can resume from last checkpoint

**6. Validation Fallbacks** (`extraction_scraper.py:110-173`)
- **Issue**: LLM may return incomplete discovered entities
- **Solution**: Strict validation checks all required fields before saving
- **Filtering**: Invalid entities silently dropped with warning log
- **Example**: Discovered entity without URL is skipped

**7. Cache Protection**
- **Issue**: Corrupted cache entries
- **Solution**: JSON parsing errors caught, cache entry deleted, entity reprocessed
- **Mechanism**: `FileCache` class handles all cache I/O with error recovery

**Error Handling Philosophy**:
- ✅ **Silent recovery** for expected issues (ScrapegraphAI JSON errors)
- ✅ **Graceful degradation** for missing data (empty lists instead of crashes)
- ✅ **Automatic retries** for transient failures (Instructor max_retries)
- ✅ **Fail-fast** only for critical issues (model not found, Ollama unreachable)
- ✅ **Detailed logging** for debugging (errors.log with full stack traces)

---

## Development Notes

### Adding New Ecosystem Roles

Edit `models/entity.py` and `config/prompts.py`:
```python
class EcosystemRole(str, Enum):
    # Add new role
    NEW_ROLE = "New Role Name"
```

### Modifying Clustering

Edit `config/scrape_config.yaml`:
```yaml
relationship_analysis:
  clustering:
    capability_clusters: 20  # Increase for more granularity
```

### Customizing Relationship Types

Edit `scrapers/relationship_analyzer.py`:
- Modify `discover_relationship_types_from_clusters()` method
- Add new matching logic between clusters

---

## References

- **ScrapegraphAI**: Web scraping with LLM intelligence
- **Ollama**: Local LLM inference
- **Pydantic**: Data validation and structured output
- **Nominatim**: Geocoding service

---

**End of Documentation**

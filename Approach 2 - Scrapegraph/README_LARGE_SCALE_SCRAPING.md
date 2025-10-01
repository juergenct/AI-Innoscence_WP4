# Hamburg CE Ecosystem - Large Scale Scraping Guide

This guide explains how to run the large-scale scraping pipeline to process ~46,000+ entities from multiple data sources.

## Overview

The pipeline processes entities in three stages:
1. **Verification**: Checks if entities are Hamburg-based and Circular Economy-related
2. **Extraction**: Extracts detailed information for verified entities
3. **Geocoding & Mapping**: Geocodes addresses and builds relationship networks

## Data Sources

The following CSV files will be processed:

| Source | Entities | Column Mapping |
|--------|----------|----------------|
| `crunchbase_companies_DEU_Hamburg.csv` | ~4,800 | name → `name`, homepage_url → `website` |
| `stakeholders_hamburg.csv` | ~70 | institution → `name`, website → `website` |
| `hamburg_branchenbuch_companies_*.csv` | ~40,000 | name → `name`, website → `website` |
| `tuhh_institutes_*.csv` | ~100 | name → `name`, url → `website` |
| `openalex_institution_directory_*.csv` | ~1,600 | institution_display_name → `name`, inst_homepage_url → `website` |

**Total**: ~46,570 entities (before deduplication)

## Prerequisites

### 1. System Requirements

- Python 3.10+
- At least 8GB RAM (16GB recommended)
- 5-10GB free disk space
- Stable internet connection

### 2. Dependencies

Install required Python packages:

```bash
cd "Approach 2 - Scrapegraph"
pip install -r requirements.txt
```

Required packages:
- scrapegraphai
- pandas
- pydantic
- requests
- beautifulsoup4
- geopy
- pyyaml

### 3. Ollama Setup

The scraper uses Ollama for LLM-based verification and extraction.

**Install Ollama** (if not already installed):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Pull the required model**:
```bash
ollama pull llama3.2:3b
```

**Start Ollama server**:
```bash
ollama serve
```

Keep this running in a separate terminal during scraping.

## Step-by-Step Instructions

### Step 1: Preprocess Data Sources

Run the preprocessing script to unify all CSV files:

```bash
cd "Approach 2 - Scrapegraph"
python preprocess_entity_sources.py
```

This will:
- Load all CSV files from the base directory
- Normalize URLs and deduplicate by URL
- Create `hamburg_ce_ecosystem/data/input/entity_urls.json`
- Create `hamburg_ce_ecosystem/data/input/entity_metadata.json` (detailed metadata)

**Expected output**:
```
Hamburg CE Ecosystem - Entity Source Preprocessor
==============================================================

Base directory: /home/thiesen/Documents/AI-Innoscence_Ecosystem

✓ Loaded 4,800 entities from Crunchbase
✓ Loaded 67 entities from stakeholders_hamburg.csv
✓ Loaded 39,508 entities from Branchenbuch
✓ Loaded 97 entities from TUHH institutes
✓ Loaded 1,607 entities from OpenAlex

✓ Total entities loaded (before deduplication): 46,079

Deduplicating entities by URL...

==============================================================
STATISTICS
==============================================================

Entities by source:
  branchenbuch             : 39,508
  crunchbase               :  4,800
  openalex                 :  1,607
  tuhh_institutes          :     97
  stakeholders_hamburg     :     67

Total unique URLs: 46,000
Entities from multiple sources: 79
==============================================================

✓ Created hamburg_ce_ecosystem/data/input/entity_urls.json
✓ Created hamburg_ce_ecosystem/data/input/entity_metadata.json (detailed metadata)
✓ Total unique URLs: 46,000

✅ Preprocessing complete!
```

### Step 2: Review Configuration

Check the scraper configuration in `hamburg_ce_ecosystem/config/scrape_config.yaml`:

```yaml
llm:
  api_type: 'ollama'
  model: 'llama3.2:3b'
  temperature: 0.1
  base_url: 'http://localhost:11434'

scraper:
  headless: true
  timeout: 30
  verbose: false
  max_retries: 3
```

**Adjust if needed**:
- Change `model` if you want to use a different LLM
- Increase `timeout` for slow connections
- Set `verbose: true` for detailed debugging

### Step 3: Run the Large-Scale Scraping

Start the scraping pipeline:

```bash
cd "Approach 2 - Scrapegraph"
python run_large_scale_scraping.py
```

The script will:
1. Check that Ollama is running
2. Verify all required files exist
3. Display configuration and estimated runtime
4. Ask for confirmation before starting

**Example output**:
```
======================================================================
          HAMBURG CE ECOSYSTEM - LARGE SCALE SCRAPER
======================================================================

Running pre-flight checks...

✓ Ollama is running
✓ Required files found

Configuration:
  - Input file: hamburg_ce_ecosystem/data/input/entity_urls.json
  - Total entities: 46,000
  - Max workers: 5
  - Estimated runtime: 76h 40m
  - Start time: 2025-10-01 14:30:00

⚠️  This will process a large number of entities and may take several hours.
Do you want to proceed? [y/N]: y
```

### Step 4: Monitor Progress

The scraper saves progress periodically:

**During verification** (Stage 1):
- Results saved every 10 entities
- Location: `hamburg_ce_ecosystem/data/verified/verification_results.csv`
- Database: `hamburg_ce_ecosystem/data/final/ecosystem.db`

**During extraction** (Stage 2):
- Results saved every 10 entities  
- Location: `hamburg_ce_ecosystem/data/extracted/entity_profiles.json`
- Database: `hamburg_ce_ecosystem/data/final/ecosystem.db`

**Monitor logs**:
```bash
tail -f "Approach 2 - Scrapegraph/hamburg_ce_ecosystem/logs/scraping_errors.log"
```

### Step 5: Resume if Interrupted

If the scraping is interrupted (Ctrl+C or system crash), you can safely resume:

```bash
python run_large_scale_scraping.py
```

The scraper uses caching and database persistence, so it will skip already-processed URLs.

## Performance Tuning

### Adjust Concurrency

Edit `run_large_scale_scraping.py` to change the number of parallel workers:

```python
max_workers = 10  # Increase for faster processing (uses more resources)
```

**Recommendations**:
- **5 workers**: Safe default (2-3 days for 46k entities)
- **10 workers**: Fast but resource-intensive (1-2 days)
- **20+ workers**: May cause rate limiting or memory issues

### Adjust Rate Limiting

Edit `hamburg_ce_ecosystem/scrapers/batch_processor.py`:

```python
self.rate_limiter = DomainRateLimiter(max_per_second=5.0)  # Increase rate
```

**Warning**: Higher rates may trigger anti-bot measures on target websites.

## Expected Runtime

**Rough estimates** (based on 46,000 entities):

| Workers | Avg Time/Entity | Total Time |
|---------|-----------------|------------|
| 5       | 30 seconds      | ~77 hours  |
| 10      | 30 seconds      | ~38 hours  |
| 20      | 30 seconds      | ~19 hours  |

**Note**: Actual time varies based on:
- Network speed
- Website response times
- LLM inference speed
- System resources

## Output Files

After completion, results are saved in `hamburg_ce_ecosystem/data/final/`:

### 1. `ecosystem_entities.csv`
Detailed entity profiles in CSV format with columns:
- `entity_name`, `url`, `ecosystem_role`
- `emails`, `phone_numbers`, `contact_persons`
- `brief_description`, `ce_relation`, `ce_activities`
- `address`, `latitude`, `longitude`
- `partners`, `partner_urls`
- `extraction_timestamp`, `extraction_confidence`

### 2. `ecosystem_map.json`
Network graph structure:
```json
{
  "nodes": [
    {
      "id": "Entity Name",
      "url": "https://example.com",
      "role": "Industry Partners",
      "lat": 53.5511,
      "lon": 9.9937
    }
  ],
  "edges": [
    {
      "source": "Entity A",
      "target": "Entity B",
      "type": "partnership"
    }
  ]
}
```

### 3. `ecosystem.db`
SQLite database with three tables:
- `verification_results`: Hamburg/CE verification results
- `entity_profiles`: Detailed entity information
- `edges`: Relationship network

Query example:
```sql
SELECT entity_name, ecosystem_role, ce_relation 
FROM entity_profiles 
WHERE is_hamburg_based = 1 AND is_ce_related = 1
ORDER BY ce_confidence DESC;
```

## Troubleshooting

### Issue: "Ollama is not running"

**Solution**:
```bash
# Start Ollama in a separate terminal
ollama serve

# Or start in background
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

### Issue: "Input file not found"

**Solution**:
```bash
# Run preprocessing first
python preprocess_entity_sources.py
```

### Issue: "Too many failed requests"

**Possible causes**:
- Network issues
- Rate limiting from target sites
- Ollama overloaded

**Solutions**:
1. Reduce `max_workers` in `run_large_scale_scraping.py`
2. Increase `timeout` in `scrape_config.yaml`
3. Check network connection
4. Restart Ollama: `pkill ollama && ollama serve`

### Issue: Memory errors

**Solutions**:
1. Reduce `max_workers` (lower concurrency)
2. Restart Python process periodically
3. Close other applications

### Issue: Slow processing

**Solutions**:
1. Increase `max_workers` (if system can handle it)
2. Use a faster LLM model (e.g., `mistral:7b` instead of `llama3.2:3b`)
3. Disable geocoding temporarily (edit `batch_processor.py`)

## Advanced: Batch Processing

For extremely large datasets, process in batches:

1. **Split input file**:
```python
import json

with open('entity_urls.json') as f:
    data = json.load(f)

urls = data['Entities to Classify']
batch_size = 10000

for i in range(0, len(urls), batch_size):
    batch = urls[i:i+batch_size]
    with open(f'batch_{i//batch_size}.json', 'w') as f:
        json.dump({'Batch': batch}, f)
```

2. **Process each batch**:
```bash
for batch in batch_*.json; do
    cp $batch hamburg_ce_ecosystem/data/input/entity_urls.json
    python run_large_scale_scraping.py
done
```

## Questions?

For issues or questions, check:
- Logs: `hamburg_ce_ecosystem/logs/scraping_errors.log`
- Progress: `hamburg_ce_ecosystem/data/verified/verification_results.csv`
- Database: `hamburg_ce_ecosystem/data/final/ecosystem.db`

Good luck with your large-scale scraping! 🚀


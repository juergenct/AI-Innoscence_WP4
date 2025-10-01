# circular_scraper/pipelines.py
"""
Data processing pipelines for scraped content
Handles validation, text extraction, and data storage
"""

import re
import json
import logging
import hashlib
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from bs4 import BeautifulSoup
import trafilatura
from scrapy.exceptions import DropItem

from circular_scraper.items import CircularEconomyItem, LinkItem, ErrorItem


logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Validate scraped items before processing
    Ensures data quality and completeness
    """
    
    def __init__(self):
        self.seen_urls = set()
        self.stats = {
            'valid_items': 0,
            'duplicate_items': 0,
            'invalid_items': 0,
        }
    
    def process_item(self, item, spider):
        """Validate and clean the item"""
        
        # Only validate content items; pass through links/errors unchanged
        if not isinstance(item, CircularEconomyItem):
            return item

        # Check if URL exists
        if not item.get('url'):
            self.stats['invalid_items'] += 1
            raise DropItem("Missing URL")
        
        # Check for duplicates
        url_hash = hashlib.md5(item['url'].encode()).hexdigest()
        if url_hash in self.seen_urls:
            self.stats['duplicate_items'] += 1
            raise DropItem(f"Duplicate URL: {item['url']}")
        
        self.seen_urls.add(url_hash)
        
        # Add timestamp if missing
        if not item.get('scraped_at'):
            item['scraped_at'] = datetime.now().isoformat()
        
        # Ensure domain is set
        if not item.get('domain'):
            item['domain'] = urlparse(item['url']).netloc
        
        # Set default values
        item.setdefault('crawl_depth', 0)
        item.setdefault('internal_links', [])
        item.setdefault('external_links', [])
        item.setdefault('emails', [])
        item.setdefault('phone_numbers', [])
        
        self.stats['valid_items'] += 1
        logger.debug(f"Validated item: {item['url']}")
        
        return item
    
    def close_spider(self, spider):
        """Log statistics when spider closes"""
        logger.info(f"Validation stats: {self.stats}")


class TextExtractionPipeline:
    """
    Extract and process text content from HTML
    Uses trafilatura for high-quality text extraction
    """
    
    # Circular economy related keywords (German and English)
    CE_KEYWORDS = {
        'circular economy', 'kreislaufwirtschaft', 'recycling', 'upcycling',
        'waste management', 'abfallwirtschaft', 'sustainability', 'nachhaltigkeit',
        'resource efficiency', 'ressourceneffizienz', 'zero waste', 'null abfall',
        'reuse', 'wiederverwendung', 'refurbishment', 'aufarbeitung',
        'cradle to cradle', 'industrial symbiosis', 'industrielle symbiose',
        'urban mining', 'life cycle', 'lebenszyklus', 'biomasse', 'bioeconomy',
        'bioökonomie', 'waste-to-energy', 'abfall-zu-energie'
    }
    
    # Hamburg-related keywords
    HAMBURG_KEYWORDS = {
        'hamburg', 'hansestadt', 'elbphilharmonie', 'hafen hamburg',
        'speicherstadt', 'hafencity', 'altona', 'wandsbek', 'eimsbüttel',
        'bergedorf', 'harburg', 'tuhh', 'uni hamburg', 'haw hamburg',
        'helmut schmidt', 'hcu hamburg'
    }
    
    def __init__(self):
        self.stats = {
            'processed': 0,
            'extraction_failed': 0,
            'ce_related': 0,
            'hamburg_related': 0,
        }
        # Optional language detection
        try:
            from langdetect import detect, detect_langs, DetectorFactory  # noqa: F401
            DetectorFactory.seed = 0
            self._langdetect_available = True
        except Exception:
            self._langdetect_available = False
    
    def process_item(self, item, spider):
        """Extract text and metadata from HTML"""
        
        if not item.get('raw_html'):
            logger.warning(f"No HTML content for {item['url']}")
            return item
        
        try:
            # Use trafilatura for main content extraction
            extracted = trafilatura.extract(
                item.get('raw_html', '') or '',
                include_comments=False,
                include_tables=True,
                include_links=True,
                output_format='json',
                deduplicate=True,
                url=item.get('url') or None,
            )
            
            if extracted:
                extracted_data = json.loads(extracted)
                item['extracted_text'] = extracted_data.get('text', '')
                item['title'] = extracted_data.get('title', '')
                item['main_content'] = extracted_data.get('raw_text', '')
                
                # Extract metadata
                metadata = trafilatura.extract_metadata(item['raw_html'])
                if metadata:
                    item['meta_description'] = metadata.description
                    # Normalize language if present
                    if getattr(metadata, 'language', None):
                        item['language'] = self._normalize_language(metadata.language)
                    if metadata.author:
                        item['organization_name'] = metadata.author
            
            # Fall back to BeautifulSoup if trafilatura fails
            if not item.get('extracted_text'):
                soup = BeautifulSoup(item.get('raw_html', '') or '', 'lxml')
                item['extracted_text'] = soup.get_text(separator=' ', strip=True)
                item['title'] = soup.title.string if soup.title else ''
            
            # Extract additional information
            self._extract_contact_info(item)
            self._extract_location_info(item)
            # Determine language (normalize + fallback detection)
            self._assign_language(item)
            self._check_keywords(item)
            
            self.stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Text extraction failed for {item['url']}: {e}")
            self.stats['extraction_failed'] += 1
            item['error_message'] = str(e)
        
        return item
    
    def _extract_contact_info(self, item):
        """Extract emails, phones, and social media links"""
        text = (item.get('extracted_text') or '') + ' ' + (item.get('raw_html') or '')
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(email_pattern, text)))
        item['emails'] = [e for e in emails if not e.endswith('.png') and not e.endswith('.jpg')]
        
        # Extract phone numbers (German format)
        phone_patterns = [
            r'\+49[\s\-]?[\d\s\-]{10,}',
            r'0[\d]{2,4}[\s\-]?[\d\s\-]{5,}',
            r'\(0[\d]{2,4}\)[\s]?[\d\s\-]{5,}',
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        item['phone_numbers'] = list(set(phones))[:5]  # Limit to 5 phone numbers
        
        # Extract social media
        social_patterns = {
            'linkedin': r'linkedin\.com/(?:company|in)/[\w\-]+',
            'twitter': r'twitter\.com/[\w]+',
            'facebook': r'facebook\.com/[\w\.\-]+',
            'instagram': r'instagram\.com/[\w\.\-]+',
            'xing': r'xing\.com/(?:profile|companies)/[\w\-]+',
        }
        social_media = {}
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                social_media[platform] = matches[0]
        item['social_media'] = social_media
    
    def _extract_location_info(self, item):
        """Extract address and location information"""
        text = item.get('extracted_text', '')
        
        # Look for Hamburg postal codes (20000-21149, 22000-22769)
        postal_pattern = r'\b(2[0-1]\d{3}|22[0-7]\d{2})\b'
        postal_matches = re.findall(postal_pattern, text)
        if postal_matches:
            item['postal_code'] = postal_matches[0]
            item['city'] = 'Hamburg'
            item['country'] = 'Germany'
        
        # Look for address patterns
        address_pattern = r'([A-ZÄÖÜß][a-zäöüß]+(?:straße|strasse|weg|platz|allee|ring|damm))\s+\d+'
        address_matches = re.findall(address_pattern, text, re.IGNORECASE)
        if address_matches:
            item['address'] = address_matches[0]
    
    def _check_keywords(self, item):
        """Check for circular economy and Hamburg keywords"""
        text_lower = (
            (item.get('extracted_text') or '') + ' ' +
            (item.get('title') or '') + ' ' +
            (item.get('meta_description') or '')
        ).lower()
        
        # Check for circular economy terms
        ce_found = any(keyword in text_lower for keyword in self.CE_KEYWORDS)
        item['has_circular_economy_terms'] = ce_found
        if ce_found:
            self.stats['ce_related'] += 1
            # Extract which keywords were found
            found_keywords = [kw for kw in self.CE_KEYWORDS if kw in text_lower]
            item['keywords'] = found_keywords[:10]  # Limit to 10 keywords
        
        # Check for Hamburg references (already done in spider, but double-check)
        hamburg_found = any(keyword in text_lower for keyword in self.HAMBURG_KEYWORDS)
        if hamburg_found:
            self.stats['hamburg_related'] += 1
            # Don't override spider's Hamburg detection
            if 'has_hamburg_reference' not in item:
                item['has_hamburg_reference'] = hamburg_found

    def _normalize_language(self, code: str) -> str:
        """Normalize language codes to 'de', 'en', or 'un'."""
        if not code:
            return 'un'
        code = code.strip().lower()
        if '-' in code:
            code = code.split('-')[0]
        if code in {'de', 'en'}:
            return code
        return 'un'

    def _assign_language(self, item):
        """Assign language using existing hints and langdetect fallback."""
        current = (item.get('language') or '').strip().lower()
        current = self._normalize_language(current)
        if current in {'de', 'en'}:
            item['language'] = current
            return
        # Use extracted text primarily
        text_source = (item.get('extracted_text') or '').strip()
        if len(text_source) < 50:
            # Augment with title/meta if content is short
            aux = ((item.get('title') or '') + ' ' + (item.get('meta_description') or '')).strip()
            if len(aux) >= 20:
                text_source = (text_source + ' ' + aux).strip()
        if self._langdetect_available and text_source:
            try:
                from langdetect import detect_langs
                langs = detect_langs(text_source)
                selected = None
                if langs:
                    # Prefer de/en with probability threshold
                    for cand in langs:
                        if cand.lang.lower() in {'de', 'en'} and cand.prob >= 0.60:
                            selected = cand.lang.lower()
                            break
                    if not selected:
                        # If top is de/en even with lower prob, accept
                        top = langs[0].lang.lower()
                        if top in {'de', 'en'}:
                            selected = top
                item['language'] = selected or 'un'
            except Exception:
                item['language'] = 'un'
        else:
            # Heuristic fallback
            text_lower = text_source.lower()
            de_hits = sum(text_lower.count(w) for w in [' und ', ' der ', ' die ', ' das ', ' nicht ', ' mit '])
            en_hits = sum(text_lower.count(w) for w in [' and ', ' the ', ' this ', ' that ', ' not ', ' with '])
            if max(de_hits, en_hits) >= 2:
                item['language'] = 'de' if de_hits >= en_hits else 'en'
            else:
                item['language'] = 'un'
    
    def close_spider(self, spider):
        """Log statistics when spider closes"""
        logger.info(f"Text extraction stats: {self.stats}")


class LLMClassificationPipeline:
    """
    Real-time LLM classification during crawling
    Classifies entities for Hamburg relevance, CE relevance, and ecosystem role
    """
    
    def __init__(self):
        self.classification_cache = {}
        self.entity_samples = {}  # Collect samples per entity
        self._batch_map = {}
        self._last_flush_ts = time.time()
        self.stats = {
            'entities_classified': 0,
            'hamburg_relevant': 0,
            'ce_relevant': 0,
            'both_relevant': 0,
        }
        
        # Initialize LLM classifier (lazy loading)
        self.classifier = None
        self.enabled = False
        self.batch_size = 100
        self.min_samples = 1
        self.flush_interval_secs = 30
        self.translate_to_en = True
        self.translate_max_chars = 1000
        self.translate_model = None
        
    def open_spider(self, spider):
        """Initialize LLM classifier when spider opens"""
        try:
            from circular_scraper.utils.llm_classifier import LLMClassifier
            # Ensure debug directory exists and is set
            debug_dir = Path('data/exports') / 'llm_debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            self.classifier = LLMClassifier(debug_dir=str(debug_dir))
            if self.classifier.check_server():
                self.enabled = True
                logger.info("LLM Classification Pipeline enabled")
            else:
                logger.warning("LLM server not available. Classification disabled.")
            # Settings (robust retrieval)
            s = spider.crawler.settings
            self.batch_size = int(s.get('LLM_CLASSIFY_BATCH_SIZE', 100))
            self.min_samples = int(s.get('LLM_MIN_SAMPLES_PER_ENTITY', 1))
            self.flush_interval_secs = int(s.get('LLM_CLASSIFY_FLUSH_INTERVAL_SECS', 30))
            self.translate_to_en = bool(s.get('LLM_TRANSLATE_TO_EN', True))
            self.translate_max_chars = int(s.get('LLM_TRANSLATE_MAX_CHARS', 1000))
            self.translate_model = s.get('LLM_TRANSLATE_MODEL', None)
        except ImportError:
            logger.warning("LLM classifier not available. Skipping LLM classification.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM classifier: {e}")
    
    def process_item(self, item, spider):
        """Classify entity in real-time"""
        
        if not self.enabled or not isinstance(item, CircularEconomyItem):
            return item
        
        entity_id = item.get('entity_id')
        if not entity_id:
            return item
        
        # Collect samples for this entity (cap at 3)
        if entity_id not in self.entity_samples:
            self.entity_samples[entity_id] = []
        
        if len(self.entity_samples[entity_id]) < 3:
            sample = {
                'url': item.get('url'),
                'title': item.get('title'),
                'meta_description': item.get('meta_description'),
                'extracted_text': item.get('extracted_text', '')[:1000],  # First 1000 chars
                'language': item.get('language')
            }
            self.entity_samples[entity_id].append(sample)
        
        # Check if already classified
        if entity_id in self.classification_cache:
            result = self.classification_cache[entity_id]
            item['hamburg_llm'] = result.get('hamburg_related', False)
            item['ce_llm'] = result.get('ce_related', False)
            item['role_llm'] = result.get('role', 'Unknown')
            item['classification_confidence'] = result.get('confidence', 0.0)
            return item
        
        # Enqueue for batched classification when we have enough samples
        if len(self.entity_samples[entity_id]) >= self.min_samples and entity_id not in self._batch_map:
            ctx = {
                'entity_id': entity_id,
                'entity_name': item.get('entity_name'),
                'entity_root_url': item.get('entity_root_url'),
                'domain': item.get('domain'),
                'samples': self.entity_samples[entity_id]
            }
            self._batch_map[entity_id] = ctx

        # Flush batch when size/interval exceeded
        now = time.time()
        if len(self._batch_map) >= self.batch_size or (now - self._last_flush_ts) >= self.flush_interval_secs:
            self._flush_batch(spider)
        
        return item
    
    def _flush_batch(self, spider):
        if not self._batch_map:
            return
        contexts = list(self._batch_map.values())
        self._batch_map.clear()
        self._last_flush_ts = time.time()
        try:
            # Optional translation step to English
            if self.translate_to_en and self.classifier:
                base = self.classifier.base
                model = self.translate_model or self.classifier.model
                for ctx in contexts:
                    for s in ctx.get('samples', []):
                        lang = (s.get('language') or '').strip().lower()
                        if lang and lang != 'en':
                            text = (s.get('extracted_text') or '')[: self.translate_max_chars]
                            if not text:
                                continue
                            try:
                                payload = {
                                    "model": model,
                                    "messages": [
                                        {"role": "system", "content": "You are a translator. Translate the user's text to English. Return only the translated plain text (no comments, no code fences)."},
                                        {"role": "user", "content": text}
                                    ],
                                    "stream": False,
                                }
                                resp = requests.post(f"{base}/api/chat", json=payload, timeout=(self.classifier.connect_timeout, self.classifier.timeout))
                                resp.raise_for_status()
                                data = resp.json()
                                content = (data.get("message", {}) or {}).get("content", "").strip()
                                if content:
                                    s['extracted_text'] = content[: self.translate_max_chars]
                                    s['language'] = 'en'
                            except Exception as e:
                                logger.warning(f"Translation failed; using original text. err={e}")

            # Parallel classification using the classifier's worker count
            with ThreadPoolExecutor(max_workers=self.classifier.max_workers if self.classifier else 2) as ex:
                future_to_ctx = {ex.submit(self.classifier.classify_entity, ctx): ctx for ctx in contexts}
                for future in as_completed(future_to_ctx):
                    ctx = future_to_ctx[future]
                    entity_id = ctx.get('entity_id')
                    try:
                        result = future.result(timeout=self.classifier.timeout + self.classifier.connect_timeout + 5 if self.classifier else 40)
                        # Cache result
                        self.classification_cache[entity_id] = {
                            'hamburg_related': result.hamburg_related,
                            'ce_related': result.ce_related,
                            'role': result.role,
                            'confidence': result.confidence
                        }
                        # Update spider gates immediately for runtime decisions
                        if hasattr(spider, 'entity_ce_relevance'):
                            spider.entity_ce_relevance[entity_id] = result.ce_related
                        # If entity becomes Hamburg+CE, mark comprehensive
                        if hasattr(spider, 'entity_relevance') and hasattr(spider, 'comprehensive_entities'):
                            is_hh = bool(spider.entity_relevance.get(entity_id, False) or result.hamburg_related)
                            if is_hh and result.ce_related:
                                if entity_id not in spider.comprehensive_entities:
                                    spider.comprehensive_entities.add(entity_id)
                                    spider.stats['comprehensive_entities'] = spider.stats.get('comprehensive_entities', 0) + 1
                        # Stats
                        self.stats['entities_classified'] += 1
                        if result.hamburg_related:
                            self.stats['hamburg_relevant'] += 1
                        if result.ce_related:
                            self.stats['ce_relevant'] += 1
                        if result.hamburg_related and result.ce_related:
                            self.stats['both_relevant'] += 1
                        logger.info(f"LLM classified {entity_id}: Hamburg={result.hamburg_related}, CE={result.ce_related}, Role={result.role}")
                    except Exception as e:
                        logger.error(f"LLM batch classification failed for {entity_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to flush LLM batch: {e}")

    def close_spider(self, spider):
        """Log statistics when spider closes"""
        # Final flush
        try:
            self._flush_batch(spider)
        except Exception as e:
            logger.error(f"Final LLM batch flush failed: {e}")
        logger.info(f"LLM classification stats: {self.stats}")
        
        # Save classification results to file
        if self.classification_cache:
            output_file = Path('data/exports') / f"llm_classifications_{datetime.now():%Y%m%d_%H%M%S}.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(self.classification_cache, f, indent=2)
            logger.info(f"Saved LLM classifications to {output_file}")


class DataStoragePipeline:
    """
    Store scraped data to files (CSV, Parquet, JSON)
    Implements incremental saving and data persistence
    """
    
    def __init__(self, export_dir='data/exports'):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data buffers
        self.items_buffer = []
        self.links_buffer = []
        self.errors_buffer = []
        
        # Buffer settings
        self.buffer_size = 50  # Save every 50 items
        self.file_counter = 0
        
        # Create subdirectories
        self.csv_dir = self.export_dir / 'csv'
        self.parquet_dir = self.export_dir / 'parquet'
        self.json_dir = self.export_dir / 'json'
        
        for dir_path in [self.csv_dir, self.parquet_dir, self.json_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session timestamp
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"DataStoragePipeline initialized with session {self.session_id}")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline from crawler settings"""
        export_dir = crawler.settings.get('EXPORT_DIR', 'data/exports')
        return cls(export_dir=export_dir)
    
    def process_item(self, item, spider):
        """Add item to buffer and save if needed"""
        
        if isinstance(item, CircularEconomyItem):
            self.items_buffer.append(item.get_export_dict())
            
            # Save raw HTML separately
            self._save_raw_html(item)
            
            # Save text content
            self._save_text_content(item)
            
        elif isinstance(item, LinkItem):
            self.links_buffer.append(dict(item))
            
        elif isinstance(item, ErrorItem):
            self.errors_buffer.append(dict(item))
        
        # Check if we should flush buffers
        if len(self.items_buffer) >= self.buffer_size:
            self._flush_buffers()
        
        return item
    
    def _save_raw_html(self, item):
        """Save raw HTML organized by entity"""
        if item.get('raw_html'):
            # Organize by entity for better structure
            entity_id = item.get('entity_id', 'unknown').replace('/', '_').replace('.', '_')
            entity_name = re.sub(r'[^\w\s-]', '', item.get('entity_name', entity_id))[:50]
            
            # Create entity-specific directory
            html_dir = self.export_dir.parent / 'raw' / self.session_id / 'entities' / f"{entity_name}_{entity_id}"
            html_dir.mkdir(parents=True, exist_ok=True)
            
            # Create descriptive filename
            url_hash = hashlib.md5(item['url'].encode()).hexdigest()[:8]
            page_title = re.sub(r'[^\w\s-]', '', item.get('title', 'page'))[:50].strip()
            if not page_title:
                page_title = 'page'
            
            filename = f"{page_title}_{url_hash}.html"
            html_path = html_dir / filename
            
            # Save HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(item['raw_html'])
            
            # Save metadata alongside HTML
            meta_filename = f"{page_title}_{url_hash}_meta.json"
            meta_path = html_dir / meta_filename
            
            metadata = {
                'url': item['url'],
                'title': item.get('title', ''),
                'entity_id': item.get('entity_id', ''),
                'entity_name': item.get('entity_name', ''),
                'scraped_at': item.get('scraped_at', ''),
                'has_hamburg_reference': item.get('has_hamburg_reference', False),
                'has_circular_economy_terms': item.get('has_circular_economy_terms', False),
                'hamburg_llm': item.get('hamburg_llm', False),
                'ce_llm': item.get('ce_llm', False),
                'role_llm': item.get('role_llm', 'Unknown'),
                'language': item.get('language', ''),
                'crawl_depth': item.get('crawl_depth', 0),
                'parent_url': item.get('parent_url', ''),
                'emails': item.get('emails', []),
                'phone_numbers': item.get('phone_numbers', [])
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved HTML and metadata for {item['url']} to {html_dir}")
    
    def _save_text_content(self, item):
        """Save extracted text content organized by entity"""
        if item.get('extracted_text'):
            # Organize by entity
            entity_id = item.get('entity_id', 'unknown').replace('/', '_').replace('.', '_')
            entity_name = re.sub(r'[^\w\s-]', '', item.get('entity_name', entity_id))[:50]
            
            # Create entity-specific directory
            text_dir = self.export_dir.parent / 'processed' / self.session_id / 'entities' / f"{entity_name}_{entity_id}"
            text_dir.mkdir(parents=True, exist_ok=True)
            
            # Create descriptive filename
            url_hash = hashlib.md5(item['url'].encode()).hexdigest()[:8]
            page_title = re.sub(r'[^\w\s-]', '', item.get('title', 'page'))[:50].strip()
            if not page_title:
                page_title = 'page'
            
            filename = f"{page_title}_{url_hash}.txt"
            text_path = text_dir / filename
            
            # Write structured text content
            with open(text_path, 'w', encoding='utf-8') as f:
                # Header with metadata
                f.write("=" * 80 + "\n")
                f.write(f"URL: {item['url']}\n")
                f.write(f"Title: {item.get('title', 'N/A')}\n")
                f.write(f"Entity: {item.get('entity_name', 'Unknown')} ({item.get('entity_id', 'unknown')})\n")
                f.write(f"Organization: {item.get('organization_name', 'N/A')}\n")
                f.write(f"Location: {item.get('city', 'N/A')}\n")
                f.write(f"Language: {item.get('language', 'unknown')}\n")
                f.write(f"Has CE Terms: {item.get('has_circular_economy_terms', False)}\n")
                f.write(f"Has Hamburg Ref: {item.get('has_hamburg_reference', False)}\n")
                
                # LLM classification if available
                if 'hamburg_llm' in item:
                    f.write(f"Hamburg (LLM): {item.get('hamburg_llm', False)}\n")
                if 'ce_llm' in item:
                    f.write(f"CE (LLM): {item.get('ce_llm', False)}\n")
                if 'role_llm' in item:
                    f.write(f"Role (LLM): {item.get('role_llm', 'Unknown')}\n")
                
                # Contact info if available
                if item.get('emails'):
                    f.write(f"Emails: {', '.join(item['emails'][:3])}\n")  # Limit to 3
                if item.get('phone_numbers'):
                    f.write(f"Phones: {', '.join(item['phone_numbers'][:3])}\n")  # Limit to 3
                
                f.write("=" * 80 + "\n\n")
                
                # Main content
                f.write(item['extracted_text'])
                
            # Also create a summary file per entity
            summary_path = text_dir / '_entity_summary.json'
            
            # Load existing summary or create new
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    entity_summary = json.load(f)
            else:
                entity_summary = {
                    'entity_id': item.get('entity_id', ''),
                    'entity_name': item.get('entity_name', ''),
                    'entity_root_url': item.get('entity_root_url', ''),
                    'pages_scraped': [],
                    'total_pages': 0,
                    'has_hamburg_reference': False,
                    'has_circular_economy_terms': False,
                    'hamburg_llm': False,
                    'ce_llm': False,
                    'role_llm': 'Unknown',
                    'all_emails': [],
                    'all_phones': [],
                    'languages': []
                }
            
            # Update summary
            entity_summary['pages_scraped'].append({
                'url': item['url'],
                'title': item.get('title', ''),
                'scraped_at': item.get('scraped_at', '')
            })
            entity_summary['total_pages'] += 1
            entity_summary['has_hamburg_reference'] = entity_summary['has_hamburg_reference'] or item.get('has_hamburg_reference', False)
            entity_summary['has_circular_economy_terms'] = entity_summary['has_circular_economy_terms'] or item.get('has_circular_economy_terms', False)
            
            # Update LLM results if available
            if 'hamburg_llm' in item:
                entity_summary['hamburg_llm'] = entity_summary.get('hamburg_llm', False) or item['hamburg_llm']
            if 'ce_llm' in item:
                entity_summary['ce_llm'] = entity_summary.get('ce_llm', False) or item['ce_llm']
            if 'role_llm' in item and item['role_llm'] != 'Unknown':
                entity_summary['role_llm'] = item['role_llm']
            
            # Update unique values using sets for deduplication
            if item.get('emails'):
                existing_emails = set(entity_summary.get('all_emails', []))
                existing_emails.update(item['emails'])
                entity_summary['all_emails'] = list(existing_emails)
            if item.get('phone_numbers'):
                existing_phones = set(entity_summary.get('all_phones', []))
                existing_phones.update(item['phone_numbers'])
                entity_summary['all_phones'] = list(existing_phones)
            if item.get('language'):
                existing_languages = set(entity_summary.get('languages', []))
                existing_languages.add(item['language'])
                entity_summary['languages'] = list(existing_languages)
            
            # Save updated summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(entity_summary, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved text content for {item['url']} to {text_dir}")
    
    def _flush_buffers(self):
        """Save buffers to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main items
        if self.items_buffer:
            df = pd.DataFrame(self.items_buffer)
            
            # Save as CSV
            csv_path = self.csv_dir / f"entities_{self.session_id}_{self.file_counter:04d}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save as Parquet
            parquet_path = self.parquet_dir / f"entities_{self.session_id}_{self.file_counter:04d}.parquet"
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            
            # Save as JSON (for debugging)
            json_path = self.json_dir / f"entities_{self.session_id}_{self.file_counter:04d}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.items_buffer, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(self.items_buffer)} items to {csv_path.name}")
            self.items_buffer.clear()
        
        # Save links
        if self.links_buffer:
            df_links = pd.DataFrame(self.links_buffer)
            links_path = self.csv_dir / f"links_{self.session_id}_{self.file_counter:04d}.csv"
            df_links.to_csv(links_path, index=False)
            self.links_buffer.clear()
        
        # Save errors
        if self.errors_buffer:
            df_errors = pd.DataFrame(self.errors_buffer)
            errors_path = self.csv_dir / f"errors_{self.session_id}_{self.file_counter:04d}.csv"
            df_errors.to_csv(errors_path, index=False)
            self.errors_buffer.clear()
        
        self.file_counter += 1
    
    def close_spider(self, spider):
        """Final save when spider closes"""
        self._flush_buffers()
        
        # Create summary file
        summary = {
            'session_id': self.session_id,
            'spider_name': spider.name,
            'total_items': self.file_counter * self.buffer_size + len(self.items_buffer),
            'files_created': self.file_counter,
            'completed_at': datetime.now().isoformat(),
        }
        
        summary_path = self.export_dir / f"summary_{self.session_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Spider closed. Summary saved to {summary_path}")


class LinkExtractionPipeline:
    """
    Extract and process links for crawling
    Identifies potential new entities to scrape
    """
    
    def __init__(self):
        self.extracted_links = set()
        self.all_links = set()
        self.entity_links = {}  # Track which entity found each link
        self.stats = {
            'total_links': 0,
            'internal_links': 0,
            'external_links': 0,
            'potential_entities': 0,
            'hamburg_ce_entities': 0,
        }
        # Skip rules aligned with spider
        self.skip_prefixes = (
            'intranet.', 'extranet.', 'collaborating.', 'cloud.', 'login.',
            'account.', 'nextcloud.', 'studip.', 'katalog.', 'events.'
        )
        self.skip_path_fragments = (
            '/login', '/wp-admin', '/account', '/apps/forms', '/impressum', '/datenschutz'
        )
    
    def process_item(self, item, spider):
        """Extract and categorize links"""
        
        if not item.get('raw_html'):
            return item
        
        # Track entity metadata for link filtering
        entity_id = item.get('entity_id', 'unknown')
        is_hamburg = item.get('hamburg_llm', item.get('has_hamburg_reference', False))
        is_ce = item.get('ce_llm', item.get('has_circular_economy_terms', False))
        is_relevant = is_hamburg and is_ce
        
        # Initialize entity tracking if needed
        if entity_id not in self.entity_links:
            self.entity_links[entity_id] = {
                'links': set(),
                'is_hamburg': is_hamburg,
                'is_ce': is_ce,
                'is_relevant': is_relevant
            }
        else:
            # Update with LLM results if available
            if 'hamburg_llm' in item:
                self.entity_links[entity_id]['is_hamburg'] = item['hamburg_llm']
            if 'ce_llm' in item:
                self.entity_links[entity_id]['is_ce'] = item['ce_llm']
            self.entity_links[entity_id]['is_relevant'] = (
                self.entity_links[entity_id]['is_hamburg'] and 
                self.entity_links[entity_id]['is_ce']
            )
        
        soup = BeautifulSoup(item['raw_html'], 'lxml')
        base_url = item['url']
        base_domain = urlparse(base_url).netloc
        
        internal_links = []
        external_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Skip non-HTTP URLs
            if not absolute_url.startswith(('http://', 'https://')):
                continue
            
            # Categorize link
            link_domain = urlparse(absolute_url).netloc
            
            # Apply domain skip rules for iteration hygiene
            if link_domain.lower().startswith(self.skip_prefixes):
                continue

            # Apply path skip rules
            path_lower = (urlparse(absolute_url).path or '').lower()
            if any(frag in path_lower for frag in self.skip_path_fragments):
                # Allow only at depth 0 for classification pages; here we don't have depth, so skip for seeds
                continue

            # Track ALL discovered absolute links (internal + external)
            self.all_links.add(absolute_url)

            if link_domain == base_domain:
                internal_links.append(absolute_url)
                self.stats['internal_links'] += 1
            else:
                external_links.append(absolute_url)
                self.stats['external_links'] += 1
                
                # Check if this could be a potential entity
                if self._is_potential_entity(absolute_url, link.get_text()):
                    self.extracted_links.add(absolute_url)
                    self.entity_links[entity_id]['links'].add(absolute_url)
                    self.stats['potential_entities'] += 1
                    
                    if is_relevant:
                        self.stats['hamburg_ce_entities'] += 1
        
        item['internal_links'] = list(set(internal_links))[:100]  # Limit to 100
        item['external_links'] = list(set(external_links))[:100]
        
        self.stats['total_links'] += len(internal_links) + len(external_links)
        
        return item
    
    def _is_potential_entity(self, url: str, link_text: str) -> bool:
        """
        Determine if a link might point to a relevant entity
        """
        # Skip social media and common service domains
        skip_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
            'google.com', 'apple.com', 'microsoft.com', 'amazon.com',
            'wikipedia.org', 'github.com', 'stackoverflow.com',
        }
        
        domain = urlparse(url).netloc
        if any(skip in domain for skip in skip_domains):
            return False
        # Skip noisy subdomains/prefixes
        if domain.lower().startswith((
            'intranet.', 'extranet.', 'collaborating.', 'cloud.', 'login.',
            'account.', 'nextcloud.', 'studip.', 'katalog.', 'events.'
        )):
            return False
        # Skip noisy paths
        path_lower = (urlparse(url).path or '').lower()
        if any(frag in path_lower for frag in (
            '/login', '/wp-admin', '/account', '/apps/forms'
        )):
            return False
        
        # Look for Hamburg or German domains
        if '.de' in domain or 'hamburg' in domain.lower():
            return True
        
        # Check link text for relevant keywords
        text_lower = (link_text or '').lower()
        relevant_keywords = ['partner', 'kooperation', 'projekt', 'institut', 
                            'unternehmen', 'startup', 'initiative']
        
        return any(keyword in text_lower for keyword in relevant_keywords)
    
    def close_spider(self, spider):
        """Save extracted links for next crawl iteration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ALL discovered links for reference
        if self.all_links:
            all_links_file = Path('data/exports') / f"discovered_links_all_{timestamp}.txt"
            all_links_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(all_links_file, 'w') as f:
                f.write(f"# All discovered links from this crawl (internal + external)\n")
                f.write(f"# Total: {len(self.all_links)} links\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                for link in sorted(self.all_links):
                    f.write(f"{link}\n")
            
            logger.info(f"Saved {len(self.all_links)} total discovered links to {all_links_file}")
        
        # Save filtered links (only from Hamburg+CE entities) for next iteration
        relevant_links = set()
        relevant_entity_count = 0
        
        for entity_id, data in self.entity_links.items():
            # Re-evaluate relevance using spider-wide decisions if available
            hh = data.get('is_hamburg', False)
            ce = data.get('is_ce', False)
            try:
                if hasattr(spider, 'entity_relevance'):
                    hh = bool(spider.entity_relevance.get(entity_id, hh))
                if hasattr(spider, 'entity_ce_relevance'):
                    ce = bool(spider.entity_ce_relevance.get(entity_id, ce))
            except Exception:
                pass
            if hh and ce:  # Hamburg AND CE
                relevant_links.update(data['links'])
                relevant_entity_count += 1
        
        if relevant_links:
            iteration_links_file = Path('data/exports') / f"iteration_seeds_{timestamp}.txt"
            
            with open(iteration_links_file, 'w') as f:
                f.write(f"# Links from Hamburg+CE relevant entities only\n")
                f.write(f"# For use in next iteration\n")
                f.write(f"# Total: {len(relevant_links)} links from {relevant_entity_count} relevant entities\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                for link in sorted(relevant_links):
                    f.write(f"{link}\n")
            
            logger.info(f"Saved {len(relevant_links)} Hamburg+CE entity links to {iteration_links_file}")
            
            # Also create CSV for easier loading
            iteration_csv = Path('data/exports') / f"iteration_seeds_{timestamp}.csv"
            with open(iteration_csv, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=['website'])
                writer.writeheader()
                for link in sorted(relevant_links):
                    writer.writerow({'website': link})
            
            logger.info(f"Also saved iteration seeds as CSV: {iteration_csv}")
        
        # Generate link statistics report
        stats_file = Path('data/exports') / f"link_stats_{timestamp}.json"
        stats_report = {
            'timestamp': datetime.now().isoformat(),
            'total_entities': len(self.entity_links),
            'hamburg_entities': sum(1 for e in self.entity_links.values() if e['is_hamburg']),
            'ce_entities': sum(1 for e in self.entity_links.values() if e['is_ce']),
            'relevant_entities': sum(1 for e in self.entity_links.values() if e['is_relevant']),
            'all_discovered_links': len(self.all_links),
            'iteration_seed_links': len(relevant_links),
            'stats': self.stats
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        logger.info(f"Link extraction stats: {self.stats}")
        logger.info(f"Entities breakdown - Total: {stats_report['total_entities']}, "
                   f"Hamburg: {stats_report['hamburg_entities']}, "
                   f"CE: {stats_report['ce_entities']}, "
                   f"Both (relevant): {stats_report['relevant_entities']}")
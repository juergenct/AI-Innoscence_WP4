# circular_scraper/pipelines.py
"""
Data processing pipelines for scraped content
Handles validation, text extraction, and data storage
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

import pandas as pd
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
        
        # Check for Hamburg references
        hamburg_found = any(keyword in text_lower for keyword in self.HAMBURG_KEYWORDS)
        item['has_hamburg_reference'] = hamburg_found
        if hamburg_found:
            self.stats['hamburg_related'] += 1

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
        """Save raw HTML to separate file"""
        if item.get('raw_html'):
            html_dir = self.export_dir.parent / 'raw' / self.session_id
            html_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename from URL
            url_hash = hashlib.md5(item['url'].encode()).hexdigest()[:8]
            filename = f"{item.get('domain', 'unknown')}_{url_hash}.html"
            
            html_path = html_dir / filename
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(item['raw_html'])
            
            logger.debug(f"Saved HTML for {item['url']} to {html_path}")
    
    def _save_text_content(self, item):
        """Save extracted text content"""
        if item.get('extracted_text'):
            text_dir = self.export_dir.parent / 'processed' / self.session_id
            text_dir.mkdir(parents=True, exist_ok=True)
            
            url_hash = hashlib.md5(item['url'].encode()).hexdigest()[:8]
            filename = f"{item.get('domain', 'unknown')}_{url_hash}.txt"
            
            text_path = text_dir / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {item['url']}\n")
                f.write(f"Title: {item.get('title', 'N/A')}\n")
                f.write(f"Organization: {item.get('organization_name', 'N/A')}\n")
                f.write(f"Location: {item.get('city', 'N/A')}\n")
                f.write(f"Has CE Terms: {item.get('has_circular_economy_terms', False)}\n")
                f.write(f"Has Hamburg Ref: {item.get('has_hamburg_reference', False)}\n")
                f.write("-" * 80 + "\n")
                f.write(item['extracted_text'])
    
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
        self.stats = {
            'total_links': 0,
            'internal_links': 0,
            'external_links': 0,
            'potential_entities': 0,
        }
    
    def process_item(self, item, spider):
        """Extract and categorize links"""
        
        if not item.get('raw_html'):
            return item
        
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
            
            if link_domain == base_domain:
                internal_links.append(absolute_url)
                self.stats['internal_links'] += 1
            else:
                external_links.append(absolute_url)
                self.stats['external_links'] += 1
                
                # Check if this could be a potential entity
                if self._is_potential_entity(absolute_url, link.get_text()):
                    self.extracted_links.add(absolute_url)
                    self.stats['potential_entities'] += 1
        
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
        if self.extracted_links:
            links_file = Path('data/exports') / f"discovered_links_{datetime.now():%Y%m%d}.txt"
            links_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(links_file, 'w') as f:
                for link in sorted(self.extracted_links):
                    f.write(f"{link}\n")
            
            logger.info(f"Saved {len(self.extracted_links)} potential entity links to {links_file}")
        
        logger.info(f"Link extraction stats: {self.stats}")
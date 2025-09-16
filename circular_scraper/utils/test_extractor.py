# circular_scraper/utils/text_extractor.py
"""
Advanced text extraction utilities using trafilatura
Handles German and English content with high quality extraction
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import trafilatura
from bs4 import BeautifulSoup
from trafilatura.settings import use_config


logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Advanced text extraction with trafilatura
    Optimized for German and English content
    """
    
    def __init__(self):
        """Initialize extractor with optimized settings"""
        # Configure trafilatura for better extraction
        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        
        # Compile regex patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(?:\+49|0049|0)[\s\-]?(?:\d{2,4}[\s\-]?)?\d{3,10}'
        )
        self.postal_pattern = re.compile(
            r'\b(?:2[0-1]\d{3}|22[0-7]\d{2})\b'  # Hamburg postal codes
        )
    
    def extract_all(self, html: str, url: str = None) -> Dict:
        """
        Extract all relevant information from HTML
        
        Args:
            html: Raw HTML content
            url: Source URL (optional, for better extraction)
        
        Returns:
            Dictionary with extracted information
        """
        result = {
            'text': '',
            'main_content': '',
            'title': '',
            'description': '',
            'author': '',
            'date': '',
            'language': '',
            'keywords': [],
            'contacts': {},
            'location': {},
            'links': [],
            'metadata': {}
        }
        
        try:
            # Main content extraction with trafilatura
            extracted = trafilatura.extract(
                html,
                output_format='json',
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_formatting=False,
                deduplicate=True,
                target_language='de',
                url=url,
                config=self.config
            )
            
            if extracted:
                import json
                data = json.loads(extracted)
                result['text'] = data.get('text', '')
                result['main_content'] = data.get('raw_text', result['text'])
                result['title'] = data.get('title', '')
                result['author'] = data.get('author', '')
                result['date'] = data.get('date', '')
            
            # Metadata extraction
            metadata = trafilatura.extract_metadata(html, url)
            if metadata:
                result['description'] = metadata.description or ''
                result['language'] = metadata.language or ''
                result['metadata'] = {
                    'sitename': metadata.sitename,
                    'categories': metadata.categories,
                    'tags': metadata.tags,
                    'license': metadata.license,
                }
            
            # Extract contacts and location
            result['contacts'] = self.extract_contacts(html)
            result['location'] = self.extract_location(html)
            
            # Extract keywords
            result['keywords'] = self.extract_keywords(result['text'])
            
            # Extract clean links
            result['links'] = self.extract_clean_links(html, url)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            # Fallback to BeautifulSoup
            result = self._fallback_extraction(html, url)
        
        return result
    
    def extract_contacts(self, html: str) -> Dict:
        """
        Extract contact information from HTML
        
        Args:
            html: Raw HTML content
        
        Returns:
            Dictionary with emails, phones, and social media
        """
        contacts = {
            'emails': [],
            'phones': [],
            'social_media': {}
        }
        
        # Create text version for better extraction
        soup = BeautifulSoup(html, 'lxml')
        text = soup.get_text(separator=' ', strip=True)
        
        # Extract emails
        emails = self.email_pattern.findall(text)
        # Filter out images and common false positives
        contacts['emails'] = [
            email for email in emails 
            if not any(email.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf'])
            and not email.startswith('example@')
        ][:10]  # Limit to 10 emails
        
        # Extract phone numbers
        phones = self.phone_pattern.findall(text)
        # Clean and format phone numbers
        cleaned_phones = []
        for phone in phones:
            cleaned = re.sub(r'[\s\-\(\)]', '', phone)
            if 6 <= len(cleaned) <= 15:  # Valid phone length
                cleaned_phones.append(phone)
        contacts['phones'] = list(set(cleaned_phones))[:5]
        
        # Extract social media links
        social_patterns = {
            'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/(?:company|in)/[\w\-]+',
            'xing': r'(?:https?://)?(?:www\.)?xing\.com/(?:profile|companies)/[\w\-]+',
            'twitter': r'(?:https?://)?(?:www\.)?twitter\.com/[\w]+',
            'facebook': r'(?:https?://)?(?:www\.)?facebook\.com/[\w\.\-]+',
            'instagram': r'(?:https?://)?(?:www\.)?instagram\.com/[\w\.\-]+',
        }
        
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                contacts['social_media'][platform] = matches[0]
        
        return contacts
    
    def extract_location(self, text: str) -> Dict:
        """
        Extract location information, focusing on Hamburg
        
        Args:
            text: Text content to analyze
        
        Returns:
            Dictionary with location information
        """
        location = {
            'is_hamburg': False,
            'postal_codes': [],
            'districts': [],
            'addresses': []
        }
        
        # Hamburg postal codes
        postal_codes = self.postal_pattern.findall(text)
        location['postal_codes'] = list(set(postal_codes))[:3]
        
        if postal_codes:
            location['is_hamburg'] = True
        
        # Hamburg districts
        districts = [
            'Altona', 'Bergedorf', 'Eimsbüttel', 'Hamburg-Mitte',
            'Hamburg-Nord', 'Harburg', 'Wandsbek', 'HafenCity',
            'Speicherstadt', 'St. Pauli', 'Ottensen', 'Blankenese',
            'Winterhude', 'Eppendorf', 'Barmbek'
        ]
        
        text_lower = text.lower()
        found_districts = [d for d in districts if d.lower() in text_lower]
        location['districts'] = found_districts
        
        if found_districts:
            location['is_hamburg'] = True
        
        # Check for Hamburg mentions
        if 'hamburg' in text_lower:
            location['is_hamburg'] = True
        
        # Extract street addresses
        address_pattern = r'([A-ZÄÖÜß][a-zäöüß]+(?:straße|strasse|weg|platz|allee|ring|damm|chaussee|kai|deich))\s+\d+[a-z]?'
        addresses = re.findall(address_pattern, text, re.IGNORECASE)
        location['addresses'] = list(set(addresses))[:5]
        
        return location
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract relevant keywords from text
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
        
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Circular economy related terms
        ce_terms = {
            'kreislaufwirtschaft', 'circular economy', 'recycling',
            'nachhaltigkeit', 'sustainability', 'ressourceneffizienz',
            'resource efficiency', 'abfallwirtschaft', 'waste management',
            'wiederverwendung', 'reuse', 'upcycling', 'zero waste',
            'bioökonomie', 'bioeconomy', 'urban mining', 'cradle to cradle'
        }
        
        # Find which terms appear in text
        text_lower = text.lower()
        found_keywords = []
        
        for term in ce_terms:
            if term in text_lower:
                # Count occurrences
                count = text_lower.count(term)
                found_keywords.append((term, count))
        
        # Sort by frequency
        found_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the terms
        return [term for term, _ in found_keywords[:max_keywords]]
    
    def extract_clean_links(self, html: str, base_url: str = None) -> List[Dict]:
        """
        Extract and categorize links from HTML
        
        Args:
            html: Raw HTML content
            base_url: Base URL for resolving relative links
        
        Returns:
            List of link dictionaries
        """
        links = []
        soup = BeautifulSoup(html, 'lxml')
        
        seen_urls = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Skip empty or anchor links
            if not href or href.startswith('#'):
                continue
            
            # Resolve relative URLs
            if base_url and not href.startswith(('http://', 'https://')):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
            
            # Skip if already seen
            if href in seen_urls:
                continue
            seen_urls.add(href)
            
            # Categorize link
            link_type = self._categorize_link(href, text)
            
            if link_type != 'skip':
                links.append({
                    'url': href,
                    'text': text[:100],  # Limit text length
                    'type': link_type
                })
        
        return links[:200]  # Limit total links
    
    def _categorize_link(self, url: str, text: str) -> str:
        """
        Categorize a link based on URL and text
        
        Args:
            url: Link URL
            text: Link text
        
        Returns:
            Link category or 'skip' to ignore
        """
        url_lower = url.lower()
        text_lower = text.lower()
        
        # Skip common irrelevant links
        skip_patterns = [
            'javascript:', 'mailto:', 'tel:', '.pdf', '.doc',
            'download', 'cookie', 'privacy', 'datenschutz', 'impressum'
        ]
        
        if any(pattern in url_lower for pattern in skip_patterns):
            return 'skip'
        
        # Categorize by content
        if any(term in url_lower + text_lower for term in 
               ['projekt', 'project', 'forschung', 'research']):
            return 'project'
        
        if any(term in url_lower + text_lower for term in 
               ['partner', 'kooperation', 'cooperation']):
            return 'partner'
        
        if any(term in url_lower + text_lower for term in 
               ['team', 'mitarbeiter', 'staff', 'kontakt', 'contact']):
            return 'contact'
        
        if any(term in url_lower + text_lower for term in 
               ['news', 'aktuell', 'presse', 'press']):
            return 'news'
        
        # Check if external
        if url.startswith(('http://', 'https://')):
            return 'external'
        
        return 'internal'
    
    def _fallback_extraction(self, html: str, url: str = None) -> Dict:
        """
        Fallback extraction using BeautifulSoup
        
        Args:
            html: Raw HTML content
            url: Source URL
        
        Returns:
            Basic extracted information
        """
        soup = BeautifulSoup(html, 'lxml')
        
        result = {
            'text': soup.get_text(separator=' ', strip=True),
            'main_content': '',
            'title': soup.title.string if soup.title else '',
            'description': '',
            'author': '',
            'date': '',
            'language': '',
            'keywords': [],
            'contacts': self.extract_contacts(html),
            'location': {},
            'links': [],
            'metadata': {}
        }
        
        # Try to get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            result['description'] = meta_desc.get('content', '')
        
        # Try to get language
        html_tag = soup.find('html')
        if html_tag:
            result['language'] = html_tag.get('lang', '')[:2]
        
        return result
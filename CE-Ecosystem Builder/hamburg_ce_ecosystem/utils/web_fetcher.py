"""Intelligent web content fetcher using Playwright for multi-page crawling."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup


def fetch_website_content(
    url: str,
    max_pages: int = 3,
    timeout: int = 30000
) -> str:
    """
    Fetch comprehensive content from website including key subsites.
    
    Uses Playwright for JavaScript rendering and intelligent navigation to:
    - Main page
    - /about, /ueber-uns pages
    - /contact, /kontakt pages  
    - /impressum page
    - /sustainability, /nachhaltigkeit pages
    
    Args:
        url: Website URL to scrape
        max_pages: Maximum number of pages to visit
        timeout: Timeout in milliseconds
    
    Returns:
        Combined text content from all visited pages
    """
    all_text = []
    visited_urls = set()
    
    # Common subpages to check (German & English)
    subpages_to_check = [
        '',  # Homepage
        '/about', '/about-us', '/ueber-uns', '/über-uns',
        '/contact', '/kontakt', '/contact-us',
        '/impressum', '/imprint', '/legal',
        '/sustainability', '/nachhaltigkeit', '/circular-economy', '/kreislaufwirtschaft'
    ]
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            )
            page = context.new_page()
            page.set_default_timeout(timeout)
            
            # Try to visit each subpage
            pages_visited = 0
            for subpath in subpages_to_check:
                if pages_visited >= max_pages:
                    break
                
                target_url = urljoin(url, subpath)
                
                # Skip if already visited or different domain
                if target_url in visited_urls:
                    continue
                if urlparse(target_url).netloc != urlparse(url).netloc:
                    continue
                
                try:
                    page.goto(target_url, wait_until='domcontentloaded', timeout=timeout)
                    visited_urls.add(target_url)
                    pages_visited += 1
                    
                    # Extract text content
                    html = page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for element in soup(['script', 'style', 'nav', 'footer']):
                        element.decompose()
                    
                    text = soup.get_text(' ', strip=True)
                    if text:
                        all_text.append(f"=== {target_url} ===\n{text}\n")
                
                except PlaywrightTimeout:
                    continue
                except Exception:
                    continue
            
            browser.close()
    
    except Exception as e:
        # If Playwright completely fails, fallback to simple fetch
        try:
            import requests
            resp = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            for element in soup(['script', 'style']):
                element.decompose()
            return soup.get_text(' ', strip=True)[:15000]
        except Exception:
            raise RuntimeError(f"Failed to fetch {url}: {e}")
    
    # Combine all text and limit size
    combined = '\n'.join(all_text)
    return combined[:15000] if combined else ""


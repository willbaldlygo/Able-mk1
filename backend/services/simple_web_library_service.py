"""Simple web library service for Able - lightweight offline web archiving."""
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimpleWebLibraryService:
    """Simple, lightweight web content archiving service."""
    
    def __init__(self):
        self.timeout = 30  # 30 seconds
        self.max_content_length = 10 * 1024 * 1024  # 10MB limit
        self.min_content_length = 100  # Minimum content length
        self.min_word_count = 10  # Minimum word count
        
        self.headers = {
            'User-Agent': 'Able-Research-Assistant/1.0 (Offline Research Tool)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_url(self, url: str, custom_title: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Scrape URL and return (title, text_content, file_path).
        Returns (None, None, None) on failure.
        """
        try:
            # Validate URL format
            if not self._is_valid_url(url):
                logger.error(f"Invalid URL format: {url}")
                return None, None, None
            
            # Fetch the webpage
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content length
            if len(response.content) > self.max_content_length:
                logger.error(f"Content too large: {len(response.content)} bytes from {url}")
                return None, None, None
            
            # Check if content is HTML
            content_type = response.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in ['text/html', 'application/xhtml']):
                logger.error(f"Non-HTML content type: {content_type} from {url}")
                return None, None, None
            
            # Extract title and clean content
            title, clean_text = self._extract_content(response.text, url, custom_title)
            
            if not clean_text or len(clean_text.strip()) < self.min_content_length:
                logger.error(f"Insufficient content extracted from URL: {url}")
                return None, None, None
            
            # Check word count
            word_count = len(clean_text.split())
            if word_count < self.min_word_count:
                logger.error(f"Insufficient word count: {word_count} from {url}")
                return None, None, None
            
            # Generate text content with metadata header
            text_content = self._generate_text_content(title, clean_text, url, word_count)
            
            # Generate file path
            safe_title = self.sanitize_filename(title)
            file_path = f"web_content_{safe_title}.txt"
            
            return title, text_content, file_path
            
        except requests.RequestException as e:
            logger.error(f"Request error for URL {url}: {str(e)}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None, None, None
    
    def _extract_content(self, html: str, url: str, custom_title: Optional[str] = None) -> Tuple[str, str]:
        """Extract title and clean text content from HTML."""
        try:
            # Use python-readability if available for better content extraction
            if READABILITY_AVAILABLE:
                doc = Document(html)
                title = custom_title or doc.title() or self._extract_title_from_url(url)
                # Get the main content and convert to text
                content_html = doc.summary()
                clean_text = self._html_to_text(content_html)
            else:
                # Fallback to BeautifulSoup only
                soup = BeautifulSoup(html, 'html.parser')
                title = custom_title or self._extract_title_from_soup(soup) or self._extract_title_from_url(url)
                clean_text = self._extract_text_from_soup(soup)
            
            return title.strip(), clean_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            # Fallback to basic BeautifulSoup extraction
            soup = BeautifulSoup(html, 'html.parser')
            title = custom_title or self._extract_title_from_soup(soup) or self._extract_title_from_url(url)
            clean_text = self._extract_text_from_soup(soup)
            return title.strip(), clean_text.strip()
    
    def _extract_title_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract title from BeautifulSoup object."""
        # Try title tag first
        title_tag = soup.find('title')
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.text.strip():
            return h1_tag.text.strip()
        
        # Try og:title meta tag
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        return ""
    
    def _extract_text_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract clean text from BeautifulSoup object."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 
                           'menu', 'advertisement', 'ads', 'sidebar', 'comment']):
            element.decompose()
        
        # Try to find main content areas first
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '.post', '.entry']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use the whole body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text and clean it
        text = main_content.get_text(separator=' ', strip=True)
        return self._clean_text(text)
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Get text and clean it
        text = soup.get_text(separator=' ', strip=True)
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove lines that are mostly punctuation or very short
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not re.match(r'^[^\w]*$', line):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def _generate_text_content(self, title: str, content: str, url: str, word_count: int) -> str:
        """Generate the final text content with metadata header."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        header = f"""=== WEB CONTENT ===
URL: {url}
Title: {title}
Saved: {timestamp}
Content Length: {word_count:,} words
===================

"""
        
        return header + content
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a reasonable title from URL if page title is unavailable."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            path = parsed.path.strip('/').replace('/', ' - ')
            
            if path:
                # Clean up common URL patterns
                path = re.sub(r'\.(html|htm|php|aspx)$', '', path, flags=re.IGNORECASE)
                path = re.sub(r'[_-]+', ' ', path)
                return f"{domain} - {path}".title()
            else:
                return domain.title()
                
        except Exception:
            return "Web Document"
    
    def sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename."""
        # Remove or replace invalid characters
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        title = re.sub(r'\s+', '_', title.strip())
        
        # Remove leading/trailing underscores or dots
        title = title.strip('_.')
        
        # Limit length
        if len(title) > 100:
            title = title[:100].rstrip('_.')
        
        # Ensure it's not empty and doesn't start with a number
        if not title or title[0].isdigit():
            title = "web_document_" + title
        
        return title
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible and safe to scrape."""
        try:
            if not self._is_valid_url(url):
                return False
            
            # Check if URL is reachable with a HEAD request
            response = requests.head(
                url, 
                headers=self.headers,
                timeout=10, 
                allow_redirects=True
            )
            return response.status_code < 400
            
        except Exception:
            return False
    
    def get_content_type(self, url: str) -> Optional[str]:
        """Get content type of URL without downloading full content."""
        try:
            response = requests.head(
                url, 
                headers=self.headers,
                timeout=10, 
                allow_redirects=True
            )
            return response.headers.get('content-type', '').lower()
        except Exception:
            return None
    
    def is_scrapable_content(self, url: str) -> bool:
        """Check if URL contains scrapable content (HTML)."""
        content_type = self.get_content_type(url)
        if not content_type:
            return True  # Assume scrapable if we can't determine
        
        # Allow HTML content
        return any(ct in content_type for ct in ['text/html', 'application/xhtml'])
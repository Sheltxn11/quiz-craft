#!/usr/bin/env python3
"""
Text Processing Module for Phase 3: Clean and Structure Text
Handles cleaning, structuring, and metadata extraction from OCR text
"""
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TextSegment:
    """Represents a structured text segment with metadata"""
    content: str
    page_number: int
    segment_type: str  # 'title', 'header', 'paragraph', 'list', 'footer'
    section_title: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextProcessor:
    """Main class for processing and structuring OCR text"""
    
    def __init__(self):
        self.common_headers = [
            r'page \d+',
            r'\d+/\d+',
            r'confidential',
            r'draft',
            r'internal use only',
            r'copyright',
            r'©',
            r'all rights reserved'
        ]
        
        self.common_footers = [
            r'page \d+ of \d+',
            r'\d+ \| page',
            r'www\.',
            r'http[s]?://',
            r'email:',
            r'tel:',
            r'phone:',
            r'fax:'
        ]
        
        # Title patterns (short lines, often capitalized or with special formatting)
        self.title_patterns = [
            r'^[A-Z][A-Z\s\d\-\.\:]{5,50}$',  # ALL CAPS titles
            r'^#{1,6}\s+.{5,100}$',  # Markdown headers
            r'^[A-Z][a-zA-Z\s\d\-\.\:]{5,80}$',  # Title case
            r'^\d+\.\s+[A-Z][a-zA-Z\s\d\-\.\:]{5,80}$',  # Numbered sections
        ]

    def process_text(self, raw_text: str, page_from: Optional[int] = None, 
                    page_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Main processing function that cleans and structures text
        
        Args:
            raw_text: Raw OCR text
            page_from: Starting page number (1-based)
            page_to: Ending page number (1-based)
            
        Returns:
            Dictionary containing structured text and metadata
        """
        start_time = time.time()
        logger.info(f"Processing text: {len(raw_text)} characters, pages {page_from}-{page_to}")

        # Step 1: Parse pages from OCR response
        pages = self._parse_pages(raw_text)
        logger.info(f"Parsed {len(pages)} pages from OCR response")
        
        # Step 2: Filter pages based on range
        if page_from is not None or page_to is not None:
            pages = self._filter_pages(pages, page_from, page_to)
            logger.info(f"Filtered to {len(pages)} pages (range: {page_from}-{page_to})")
        
        # Step 3: Clean each page
        cleaned_pages = []
        for page_num, page_content in pages.items():
            cleaned_content = self._clean_page_text(page_content)
            cleaned_pages.append((page_num, cleaned_content))
        
        # Step 4: Structure text into segments
        segments = []
        current_section = None
        
        for page_num, page_content in cleaned_pages:
            page_segments = self._structure_page_text(page_content, page_num)
            
            # Update section context
            for segment in page_segments:
                if segment.segment_type in ['title', 'header']:
                    current_section = segment.content.strip()
                else:
                    segment.section_title = current_section
                segments.append(segment)
        
        # Step 5: Generate metadata
        metadata = self._generate_metadata(segments, pages)
        
        # Step 6: Create final structured output
        structured_text = self._create_structured_output(segments, metadata)
        
        logger.info(f"Processing complete: {len(segments)} segments created")
        return structured_text

    def _parse_pages(self, raw_text: str) -> Dict[int, str]:
        """Parse pages from OCR response, handling different formats"""
        pages = {}

        logger.info(f"Parsing pages from text of length {len(raw_text)}")

        # Method 1: Try to detect our custom page markers (PAGE_X_START/END)
        if 'PAGE_' in raw_text and '_START' in raw_text and '_END' in raw_text:
            logger.info("Detected custom page markers")
            page_pattern = r'PAGE_(\d+)_START\n(.*?)\nPAGE_\d+_END'
            matches = re.findall(page_pattern, raw_text, re.DOTALL)

            if matches:
                logger.info(f"Found {len(matches)} pages using custom markers")
                for match in matches:
                    page_num = int(match[0])
                    content = match[1].strip()
                    if content:
                        pages[page_num] = content
                return pages

        # Method 2: Try to detect Mistral OCR page structure
        if 'pages=[' in raw_text or 'OCRPageObject' in raw_text:
            logger.info("Detected Mistral OCR page structure")

            # More flexible pattern to handle different OCR response formats
            patterns = [
                r'OCRPageObject\(index=(\d+),\s*markdown="([^"]*(?:\\.[^"]*)*)"',
                r'OCRPageObject\(index=(\d+),[^"]*markdown="([^"]*(?:\\.[^"]*)*)"',
                r'index=(\d+)[^"]*markdown="([^"]*(?:\\.[^"]*)*)"'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, raw_text, re.DOTALL)
                if matches:
                    logger.info(f"Found {len(matches)} pages using pattern")
                    for match in matches:
                        page_num = int(match[0]) + 1  # Convert 0-based to 1-based
                        content = match[1].replace('\\"', '"').replace('\\n', '\n')
                        pages[page_num] = content
                    break

        # Method 3: Try to extract from string representation of pages
        if not pages and 'pages=' in raw_text:
            logger.info("Attempting to parse from string representation")
            # Look for markdown content within the response
            markdown_pattern = r'markdown=["\']([^"\']*(?:\\.[^"\']*)*)["\']'
            markdown_matches = re.findall(markdown_pattern, raw_text, re.DOTALL)

            for i, content in enumerate(markdown_matches, 1):
                if content.strip():
                    clean_content = content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                    pages[i] = clean_content

        # Method 4: Try to split by common page break indicators
        if not pages:
            logger.info("Attempting to split by page break indicators")
            page_breaks = re.split(r'\n\s*(?:page\s+\d+|---+|\f|Page\s+\d+)\s*\n', raw_text, flags=re.IGNORECASE)
            for i, page_content in enumerate(page_breaks, 1):
                if page_content.strip():
                    pages[i] = page_content.strip()

        # Method 5: Try to split by form feed characters or other page separators
        if not pages:
            logger.info("Attempting to split by form feed characters")
            page_breaks = raw_text.split('\f')  # Form feed character
            if len(page_breaks) > 1:
                for i, page_content in enumerate(page_breaks, 1):
                    if page_content.strip():
                        pages[i] = page_content.strip()

        # Method 6: Look for page numbers in the text and split accordingly
        if not pages:
            logger.info("Attempting to split by detected page numbers")
            # Look for patterns like "Page 1", "1", etc. at the beginning of lines
            page_markers = list(re.finditer(r'^(?:Page\s+)?(\d+)\s*$', raw_text, re.MULTILINE | re.IGNORECASE))

            if len(page_markers) > 1:
                for i, marker in enumerate(page_markers):
                    start_pos = marker.end()
                    end_pos = page_markers[i + 1].start() if i + 1 < len(page_markers) else len(raw_text)
                    page_num = int(marker.group(1))
                    content = raw_text[start_pos:end_pos].strip()
                    if content:
                        pages[page_num] = content

        # Fallback: treat entire text as page 1
        if not pages:
            logger.warning("No page structure detected, treating as single page")
            pages[1] = raw_text

        logger.info(f"Successfully parsed {len(pages)} pages")
        return pages

    def _filter_pages(self, pages: Dict[int, str], page_from: Optional[int], 
                     page_to: Optional[int]) -> Dict[int, str]:
        """Filter pages based on specified range"""
        if page_from is None:
            page_from = 1
        if page_to is None:
            page_to = max(pages.keys()) if pages else 1
            
        filtered_pages = {}
        for page_num, content in pages.items():
            if page_from <= page_num <= page_to:
                filtered_pages[page_num] = content
                
        return filtered_pages

    def _clean_page_text(self, text: str) -> str:
        """Clean individual page text by removing headers, footers, and noise"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip headers and footers
            if self._is_header_or_footer(line):
                continue
                
            # Clean up common OCR artifacts
            line = self._clean_line(line)
            
            if line:  # Only add non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _is_header_or_footer(self, line: str) -> bool:
        """Check if line is likely a header or footer"""
        line_lower = line.lower().strip()
        
        # Check against common header/footer patterns
        for pattern in self.common_headers + self.common_footers:
            if re.search(pattern, line_lower):
                return True
                
        # Very short lines at start/end are often headers/footers
        if len(line.strip()) < 5:
            return True
            
        # Lines with only numbers, dates, or page references
        if re.match(r'^[\d\s\-\/\.\|]+$', line.strip()):
            return True
            
        return False

    def _clean_line(self, line: str) -> str:
        """Clean individual line of common OCR artifacts"""
        # Remove excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        # Fix common OCR errors
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # Add space between camelCase
        line = re.sub(r'(\w)(\d)', r'\1 \2', line)  # Space between word and number
        line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', line)  # Space between number and word
        
        # Remove standalone special characters
        line = re.sub(r'\s+[^\w\s]\s+', ' ', line)
        
        return line.strip()

    def _structure_page_text(self, text: str, page_num: int) -> List[TextSegment]:
        """Structure page text into segments with metadata"""
        segments = []
        lines = text.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # End current paragraph
                if current_paragraph:
                    content = ' '.join(current_paragraph)
                    segment_type = self._classify_text_type(content)
                    segments.append(TextSegment(
                        content=content,
                        page_number=page_num,
                        segment_type=segment_type
                    ))
                    current_paragraph = []
                continue
            
            # Check if line is a title/header
            if self._is_title_or_header(line):
                # End current paragraph first
                if current_paragraph:
                    content = ' '.join(current_paragraph)
                    segments.append(TextSegment(
                        content=content,
                        page_number=page_num,
                        segment_type='paragraph'
                    ))
                    current_paragraph = []
                
                # Add title/header
                segments.append(TextSegment(
                    content=line,
                    page_number=page_num,
                    segment_type='title' if self._is_main_title(line) else 'header'
                ))
            else:
                current_paragraph.append(line)
        
        # Handle remaining paragraph
        if current_paragraph:
            content = ' '.join(current_paragraph)
            segment_type = self._classify_text_type(content)
            segments.append(TextSegment(
                content=content,
                page_number=page_num,
                segment_type=segment_type
            ))
        
        return segments

    def _is_title_or_header(self, line: str) -> bool:
        """Check if line is likely a title or header"""
        # Check against title patterns
        for pattern in self.title_patterns:
            if re.match(pattern, line.strip()):
                return True
        
        # Short lines (likely titles)
        if 5 <= len(line.strip()) <= 80 and not line.endswith('.'):
            return True
            
        return False

    def _is_main_title(self, line: str) -> bool:
        """Check if line is a main title (vs sub-header)"""
        # Main titles are often shorter and more prominent
        return (len(line.strip()) <= 50 and 
                (line.isupper() or line.startswith('#')))

    def _classify_text_type(self, text: str) -> str:
        """Classify text segment type"""
        text = text.strip()
        
        # List items
        if re.match(r'^[\-\*\+•]\s+', text) or re.match(r'^\d+\.\s+', text):
            return 'list'
        
        # Long text is likely paragraph
        if len(text) > 100:
            return 'paragraph'
        
        # Default to paragraph
        return 'paragraph'

    def _generate_metadata(self, segments: List[TextSegment], pages: Dict[int, str]) -> Dict[str, Any]:
        """Generate metadata about the processed text"""
        metadata = {
            'total_pages': len(pages),
            'total_segments': len(segments),
            'segment_types': defaultdict(int),
            'pages_processed': list(pages.keys()),
            'sections': [],
            'word_count': 0,
            'character_count': 0
        }
        
        current_section = None
        for segment in segments:
            metadata['segment_types'][segment.segment_type] += 1
            metadata['word_count'] += len(segment.content.split())
            metadata['character_count'] += len(segment.content)
            
            if segment.segment_type in ['title', 'header']:
                current_section = segment.content
                metadata['sections'].append({
                    'title': current_section,
                    'page': segment.page_number
                })
        
        # Convert defaultdict to regular dict
        metadata['segment_types'] = dict(metadata['segment_types'])
        
        return metadata

    def _create_structured_output(self, segments: List[TextSegment], 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create final structured output"""
        return {
            'segments': [
                {
                    'content': seg.content,
                    'page_number': seg.page_number,
                    'type': seg.segment_type,
                    'section_title': seg.section_title,
                    'metadata': seg.metadata
                }
                for seg in segments
            ],
            'metadata': metadata,
            'processed_text': '\n\n'.join(seg.content for seg in segments),
            'status': 'completed'
        }

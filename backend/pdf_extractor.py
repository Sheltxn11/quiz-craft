#!/usr/bin/env python3
"""
PDF Page Extractor - Extract specific pages from PDF before OCR
This solves the efficiency issue of processing all pages when only specific pages are needed
"""
import logging
import io
import base64
from typing import Optional, Tuple
import PyPDF2

# Configure logging
logger = logging.getLogger(__name__)

class PDFPageExtractor:
    """Extract specific pages from PDF to optimize OCR processing"""
    
    def __init__(self):
        logger.info("PDFPageExtractor initialized")
    
    def extract_pages(self, pdf_data: bytes, page_from: Optional[int] = None, 
                     page_to: Optional[int] = None) -> Tuple[bytes, dict]:
        """
        Extract specific pages from PDF
        
        Args:
            pdf_data: PDF file as bytes
            page_from: Starting page number (1-based, inclusive)
            page_to: Ending page number (1-based, inclusive)
            
        Returns:
            Tuple of (extracted_pdf_bytes, metadata)
        """
        try:
            # Create PDF reader from bytes
            pdf_stream = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} total pages")
            
            # If no page range specified, return original PDF
            if page_from is None and page_to is None:
                logger.info("No page range specified, returning original PDF")
                return pdf_data, {
                    'total_pages': total_pages,
                    'extracted_pages': list(range(1, total_pages + 1)),
                    'extraction_performed': False
                }
            
            # Set default values
            if page_from is None:
                page_from = 1
            if page_to is None:
                page_to = total_pages
                
            # Validate page range
            if page_from < 1:
                page_from = 1
            if page_to > total_pages:
                page_to = total_pages
            if page_from > page_to:
                page_from, page_to = page_to, page_from
                
            logger.info(f"Extracting pages {page_from} to {page_to} from {total_pages} total pages")
            
            # Create new PDF with only requested pages
            pdf_writer = PyPDF2.PdfWriter()
            
            # Add requested pages (convert to 0-based indexing)
            pages_added = []
            for page_num in range(page_from - 1, page_to):
                if page_num < len(pdf_reader.pages):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                    pages_added.append(page_num + 1)  # Convert back to 1-based for logging
            
            logger.info(f"Added {len(pages_added)} pages to extracted PDF: {pages_added}")
            
            # Write to bytes
            output_stream = io.BytesIO()
            pdf_writer.write(output_stream)
            extracted_pdf_bytes = output_stream.getvalue()
            
            # Close streams
            output_stream.close()
            pdf_stream.close()
            
            metadata = {
                'total_pages': total_pages,
                'requested_range': f"{page_from}-{page_to}",
                'extracted_pages': pages_added,
                'pages_extracted': len(pages_added),
                'extraction_performed': True,
                'original_size_bytes': len(pdf_data),
                'extracted_size_bytes': len(extracted_pdf_bytes),
                'size_reduction_percent': round((1 - len(extracted_pdf_bytes) / len(pdf_data)) * 100, 2)
            }
            
            logger.info(f"Page extraction completed: {metadata}")
            return extracted_pdf_bytes, metadata
            
        except Exception as e:
            logger.error(f"Error extracting pages from PDF: {e}")
            # Return original PDF if extraction fails
            return pdf_data, {
                'total_pages': 0,
                'extracted_pages': [],
                'extraction_performed': False,
                'error': str(e)
            }
    
    def extract_pages_from_base64(self, pdf_base64: str, page_from: Optional[int] = None, 
                                 page_to: Optional[int] = None) -> Tuple[str, dict]:
        """
        Extract pages from base64-encoded PDF
        
        Args:
            pdf_base64: PDF as base64 string
            page_from: Starting page number (1-based)
            page_to: Ending page number (1-based)
            
        Returns:
            Tuple of (extracted_pdf_base64, metadata)
        """
        try:
            # Decode base64 to bytes
            pdf_bytes = base64.b64decode(pdf_base64)
            
            # Extract pages
            extracted_bytes, metadata = self.extract_pages(pdf_bytes, page_from, page_to)
            
            # Encode back to base64
            extracted_base64 = base64.b64encode(extracted_bytes).decode('utf-8')
            
            # Update metadata with base64 info
            metadata['original_base64_length'] = len(pdf_base64)
            metadata['extracted_base64_length'] = len(extracted_base64)
            
            logger.info(f"Base64 extraction completed. Original: {len(pdf_base64)} chars, Extracted: {len(extracted_base64)} chars")
            
            return extracted_base64, metadata
            
        except Exception as e:
            logger.error(f"Error extracting pages from base64 PDF: {e}")
            return pdf_base64, {
                'extraction_performed': False,
                'error': str(e)
            }
    
    def get_pdf_info(self, pdf_data: bytes) -> dict:
        """
        Get basic information about a PDF
        
        Args:
            pdf_data: PDF file as bytes
            
        Returns:
            Dictionary with PDF information
        """
        try:
            pdf_stream = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            total_pages = len(pdf_reader.pages)
            
            # Get metadata if available
            metadata = pdf_reader.metadata
            
            info = {
                'total_pages': total_pages,
                'file_size_bytes': len(pdf_data),
                'has_metadata': metadata is not None
            }
            
            if metadata:
                info['title'] = metadata.get('/Title', 'Unknown')
                info['author'] = metadata.get('/Author', 'Unknown')
                info['creator'] = metadata.get('/Creator', 'Unknown')
            
            pdf_stream.close()
            return info
            
        except Exception as e:
            logger.error(f"Error getting PDF info: {e}")
            return {
                'total_pages': 0,
                'file_size_bytes': len(pdf_data),
                'error': str(e)
            }

# Global instance
pdf_extractor = PDFPageExtractor()

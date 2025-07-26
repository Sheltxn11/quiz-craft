#!/usr/bin/env python3
"""
Text Chunking Module for Phase 4: Text Chunking
Handles tokenization and chunking of structured text with overlap and metadata
"""
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from transformers import AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    chunk_id: str
    content: str
    token_count: int
    page: int
    section: Optional[str] = None
    paragraph_index: int = 0
    start_char: int = 0
    end_char: int = 0
    overlap_with_previous: bool = False
    overlap_with_next: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextChunker:
    """Main class for chunking structured text into token-based chunks"""
    
    def __init__(self, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 chunk_size: int = 400, overlap_percentage: float = 0.1):
        """
        Initialize the text chunker
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            chunk_size: Target size of each chunk in tokens
            overlap_percentage: Percentage of overlap between chunks (0.1 = 10%)
        """
        self.chunk_size = chunk_size
        self.overlap_tokens = int(chunk_size * overlap_percentage)
        self.tokenizer_name = tokenizer_name
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, overlap={self.overlap_tokens} tokens")

    def _init_tokenizer(self):
        """Initialize the appropriate tokenizer"""
        try:
            if "gpt" in self.tokenizer_name.lower() or self.tokenizer_name == "tiktoken":
                # Use tiktoken for OpenAI models
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenizer_type = "tiktoken"
                logger.info("Using tiktoken tokenizer")
            else:
                # Use HuggingFace tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self.tokenizer_type = "huggingface"
                logger.info(f"Using HuggingFace tokenizer: {self.tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {self.tokenizer_name}: {e}")
            # Fallback to tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.tokenizer_type = "tiktoken"
            logger.info("Falling back to tiktoken tokenizer")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            if self.tokenizer_type == "tiktoken":
                return len(self.tokenizer.encode(text))
            else:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to word count approximation
            return len(text.split()) * 1.3  # Rough approximation

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text and return token IDs"""
        try:
            if self.tokenizer_type == "tiktoken":
                return self.tokenizer.encode(text)
            else:
                return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []

    def detokenize_tokens(self, tokens: List[int]) -> str:
        """Convert token IDs back to text"""
        try:
            if self.tokenizer_type == "tiktoken":
                return self.tokenizer.decode(tokens)
            else:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error detokenizing: {e}")
            return ""

    def chunk_structured_text(self, structured_data: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk structured text data into token-based chunks with overlap
        
        Args:
            structured_data: Output from text_processor.process_text()
            
        Returns:
            List of TextChunk objects
        """
        start_time = time.time()
        logger.info("Starting text chunking process")

        segments = structured_data.get('segments', [])
        if not segments:
            logger.warning("No segments found in structured data")
            return []

        chunks = []
        current_tokens = []
        current_text = ""
        current_metadata = {
            'page': 1,
            'section': None,
            'paragraph_index': 0
        }
        
        segment_index = 0
        char_position = 0
        
        for segment in segments:
            segment_content = segment.get('content', '').strip()
            if not segment_content:
                continue
                
            segment_tokens = self.tokenize_text(segment_content)
            segment_token_count = len(segment_tokens)
            
            # Update metadata from segment
            page = segment.get('page_number', current_metadata['page'])
            section = segment.get('section_title', current_metadata['section'])
            
            # If adding this segment would exceed chunk size, finalize current chunk
            if (len(current_tokens) + segment_token_count > self.chunk_size and 
                len(current_tokens) > 0):
                
                # Create chunk from current tokens
                chunk = self._create_chunk(
                    current_tokens, 
                    current_metadata, 
                    segment_index,
                    char_position - len(current_text),
                    char_position
                )
                chunks.append(chunk)
                
                # Prepare overlap for next chunk
                overlap_tokens = current_tokens[-self.overlap_tokens:] if len(current_tokens) > self.overlap_tokens else current_tokens
                overlap_text = self.detokenize_tokens(overlap_tokens)
                
                # Start new chunk with overlap
                current_tokens = overlap_tokens.copy()
                current_text = overlap_text
                char_position = char_position - len(current_text) + len(overlap_text)
            
            # Add current segment to chunk
            current_tokens.extend(segment_tokens)
            current_text += (" " if current_text else "") + segment_content
            current_metadata.update({
                'page': page,
                'section': section,
                'paragraph_index': segment_index
            })
            
            segment_index += 1
            char_position += len(segment_content) + (1 if current_text else 0)  # +1 for space
        
        # Handle remaining tokens
        if current_tokens:
            chunk = self._create_chunk(
                current_tokens, 
                current_metadata, 
                segment_index,
                char_position - len(current_text),
                char_position
            )
            chunks.append(chunk)
        
        # Mark overlap relationships
        self._mark_overlaps(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(segments)} segments")
        return chunks

    def _create_chunk(self, tokens: List[int], metadata: Dict[str, Any], 
                     paragraph_index: int, start_char: int, end_char: int) -> TextChunk:
        """Create a TextChunk object from tokens and metadata"""
        content = self.detokenize_tokens(tokens)
        
        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            content=content.strip(),
            token_count=len(tokens),
            page=metadata.get('page', 1),
            section=metadata.get('section'),
            paragraph_index=paragraph_index,
            start_char=start_char,
            end_char=end_char,
            metadata={
                'tokenizer': self.tokenizer_name,
                'chunk_size_target': self.chunk_size,
                'overlap_tokens': self.overlap_tokens,
                'created_at': str(uuid.uuid4())  # Timestamp placeholder
            }
        )

    def _mark_overlaps(self, chunks: List[TextChunk]):
        """Mark which chunks have overlaps with adjacent chunks"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.overlap_with_previous = True
            if i < len(chunks) - 1:
                chunk.overlap_with_next = True

    def chunk_simple_text(self, text: str, page: int = 1, section: str = None) -> List[TextChunk]:
        """
        Chunk simple text (fallback method for non-structured text)
        
        Args:
            text: Plain text to chunk
            page: Page number
            section: Section title
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Chunking simple text of length {len(text)}")
        
        tokens = self.tokenize_text(text)
        chunks = []
        
        start_idx = 0
        paragraph_index = 0
        
        while start_idx < len(tokens):
            # Determine end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Create chunk
            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                content=self.detokenize_tokens(chunk_tokens).strip(),
                token_count=len(chunk_tokens),
                page=page,
                section=section,
                paragraph_index=paragraph_index,
                start_char=start_idx,
                end_char=end_idx,
                metadata={
                    'tokenizer': self.tokenizer_name,
                    'chunk_size_target': self.chunk_size,
                    'overlap_tokens': self.overlap_tokens,
                    'simple_chunking': True
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_tokens if end_idx < len(tokens) else end_idx
            paragraph_index += 1
        
        # Mark overlaps
        self._mark_overlaps(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from simple text")
        return chunks

    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Generate statistics about the chunks"""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        pages = set(chunk.page for chunk in chunks)
        sections = set(chunk.section for chunk in chunks if chunk.section)
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'pages_covered': sorted(list(pages)),
            'sections_covered': list(sections),
            'chunks_with_overlap': sum(1 for chunk in chunks if chunk.overlap_with_previous or chunk.overlap_with_next),
            'tokenizer_used': self.tokenizer_name
        }

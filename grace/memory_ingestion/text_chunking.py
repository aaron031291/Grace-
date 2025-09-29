"""
Text chunking utilities for memory ingestion.

Splits text into chunks suitable for embedding and vector storage (1-2k tokens).
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    start_pos: int
    end_pos: int
    chunk_hash: str
    token_count: int
    metadata: Dict[str, Any]


class TextChunker:
    """Chunk text into manageable pieces for embedding."""
    
    def __init__(self, 
                 target_chunk_size: int = 1500,
                 max_chunk_size: int = 2000,
                 min_chunk_size: int = 100,
                 overlap_size: int = 200):
        """
        Initialize chunker with size parameters.
        
        Args:
            target_chunk_size: Target number of tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            overlap_size: Number of tokens to overlap between chunks
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
    def chunk_text(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text into pieces suitable for embedding.
        
        Args:
            text: Text to chunk
            source_metadata: Metadata about the source text
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        logger.info(f"Chunking text of {len(text)} characters")
        
        # Normalize the text
        text = self._normalize_text(text)
        
        # Try intelligent chunking first (by paragraphs, sentences, etc.)
        chunks = self._intelligent_chunk(text)
        
        # If intelligent chunking produces chunks that are too large, split them
        final_chunks = []
        for chunk in chunks:
            if self._estimate_token_count(chunk) > self.max_chunk_size:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            elif self._estimate_token_count(chunk) >= self.min_chunk_size:
                final_chunks.append(chunk)
            # Skip chunks that are too small (unless it's the last chunk)
        
        # Create TextChunk objects
        text_chunks = []
        for i, chunk_text in enumerate(final_chunks):
            start_pos = text.find(chunk_text)
            if start_pos == -1:
                # Fallback to position estimation
                start_pos = i * self.target_chunk_size
            
            end_pos = start_pos + len(chunk_text)
            token_count = self._estimate_token_count(chunk_text)
            chunk_hash = self._compute_chunk_hash(chunk_text)
            
            metadata = {
                'chunk_index': i,
                'total_chunks': len(final_chunks),
                'source_length': len(text),
                **(source_metadata or {})
            }
            
            text_chunks.append(TextChunk(
                content=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                chunk_hash=chunk_hash,
                token_count=token_count,
                metadata=metadata
            ))
        
        logger.info(f"Created {len(text_chunks)} chunks")
        return text_chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent chunking."""
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        return text.strip()
    
    def _intelligent_chunk(self, text: str) -> List[str]:
        """
        Chunk text intelligently by natural boundaries.
        
        Prefers:
        1. Double newlines (paragraph breaks)
        2. Single newlines  
        3. Sentence boundaries
        4. Word boundaries
        """
        chunks = []
        current_chunk = ""
        
        # Split by double newlines first (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if adding this paragraph would make chunk too large
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if (current_chunk and 
                self._estimate_token_count(potential_chunk) > self.target_chunk_size):
                # Finalize current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a large chunk into smaller pieces."""
        chunks = []
        
        # Try splitting by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if (current_chunk and 
                self._estimate_token_count(potential_chunk) > self.target_chunk_size):
                # Finalize current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have chunks that are too large, split by words
        final_chunks = []
        for chunk in chunks:
            if self._estimate_token_count(chunk) > self.max_chunk_size:
                word_chunks = self._split_by_words(chunk)
                final_chunks.extend(word_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_by_words(self, text: str) -> List[str]:
        """Split text by words as a last resort."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            # Estimate tokens (rough approximation: word count * 1.3)
            estimated_tokens = len(current_chunk) * 1.3
            
            if estimated_tokens >= self.target_chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep some overlap
                overlap_words = current_chunk[-self.overlap_size//4:] if len(current_chunk) > self.overlap_size//4 else []
                current_chunk = overlap_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a piece of text.
        
        This is a rough approximation. For more accurate counting,
        would need to use tiktoken or similar tokenizer.
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        # Adjusted for words and punctuation
        word_count = len(text.split())
        char_count = len(text)
        
        # Heuristic: mix of word-based and character-based estimation
        estimated_tokens = int(word_count * 1.3 + char_count * 0.25)
        
        return max(estimated_tokens, 1)  # At least 1 token
    
    def _compute_chunk_hash(self, text: str) -> str:
        """Compute SHA-256 hash for a text chunk."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]  # Short hash


# Global chunker instance
_chunker = None

def get_text_chunker() -> TextChunker:
    """Get global text chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = TextChunker()
    return _chunker
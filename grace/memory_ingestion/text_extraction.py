"""
Text extraction utilities for various file formats.

Converts files to plain text for processing.
"""
import logging
import mimetypes
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import hashlib
import subprocess
import re

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from various file formats."""
    
    def __init__(self):
        self.supported_types = {
            'text/plain': self._extract_text,
            'text/markdown': self._extract_text,
            'text/html': self._extract_html,
            'application/pdf': self._extract_pdf,
            'application/json': self._extract_json,
            'application/xml': self._extract_xml,
            'text/csv': self._extract_csv,
        }
    
    async def extract_text(self, file_path: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from a file.
        
        Returns:
            Dict with extracted text, metadata, and processing info
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine MIME type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(str(path))
        
        if not mime_type:
            # Try to detect by extension
            ext = path.suffix.lower()
            ext_map = {
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.html': 'text/html',
                '.pdf': 'application/pdf',
                '.json': 'application/json',
                '.xml': 'application/xml',
                '.csv': 'text/csv',
            }
            mime_type = ext_map.get(ext, 'text/plain')
        
        logger.info(f"Extracting text from {file_path} (type: {mime_type})")
        
        # Get file stats
        stat = path.stat()
        file_hash = await self._compute_file_hash(file_path)
        
        # Extract text using appropriate method
        extractor = self.supported_types.get(mime_type, self._extract_text)
        
        try:
            text_content = await extractor(file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            # Fallback to plain text
            text_content = await self._extract_text(file_path)
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text_content)
        
        return {
            'text': cleaned_text,
            'metadata': {
                'file_path': str(path),
                'file_name': path.name,
                'file_size': stat.st_size,
                'mime_type': mime_type,
                'file_hash': file_hash,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'line_count': len(cleaned_text.splitlines()),
            }
        }
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract plain text content."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    async def _extract_html(self, file_path: str) -> str:
        """Extract text from HTML files."""
        # Simple HTML tag removal for now
        text = await self._extract_text(file_path)
        # Remove HTML tags
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    async def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        # For now, return placeholder - would need PyPDF2 or similar
        return f"[PDF content from {file_path} - text extraction placeholder]"
    
    async def _extract_json(self, file_path: str) -> str:
        """Extract text from JSON files."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)
    
    async def _extract_xml(self, file_path: str) -> str:
        """Extract text from XML files."""
        try:
            from xml.etree import ElementTree as ET
            tree = ET.parse(file_path)
            return ET.tostring(tree.getroot(), encoding='unicode')
        except Exception:
            # Fallback to plain text
            return await self._extract_text(file_path)
    
    async def _extract_csv(self, file_path: str) -> str:
        """Extract text from CSV files."""
        # Simple CSV reading for now
        return await self._extract_text(file_path)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# Global extractor instance
_extractor = None

def get_text_extractor() -> TextExtractor:
    """Get global text extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = TextExtractor()
    return _extractor
"""
Hunter Protocol - Universal Data Adapters
=========================================

Adapters for processing different data types:
- CODE: Python, JavaScript, TypeScript, etc.
- DOCUMENT: PDF, Word, Markdown, Text
- MEDIA: Images (OCR), Audio (ASR), Video
- STRUCTURED: CSV, JSON, Parquet, Excel
- WEB: URLs, APIs, HTML
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataAdapter(ABC):
    """Base class for data adapters"""
    
    @abstractmethod
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process data and return structured result"""
        pass
    
    @abstractmethod
    async def validate(self, raw_data: bytes) -> bool:
        """Validate data can be processed"""
        pass


class CodeAdapter(DataAdapter):
    """Adapter for code (Python, JavaScript, TypeScript, etc.)"""
    
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process code data"""
        logger.info("CodeAdapter: Processing code...")
        
        try:
            code = raw_data.decode('utf-8')
            
            # Extract metadata
            language = self._detect_language(code, metadata)
            functions = self._extract_functions(code)
            classes = self._extract_classes(code)
            imports = self._extract_imports(code)
            
            return {
                "success": True,
                "language": language,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "line_count": len(code.split('\n')),
                "has_tests": 'test_' in code or 'assert' in code,
                "has_docs": '"""' in code or "'''" in code
            }
        except Exception as e:
            logger.error(f"CodeAdapter error: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, raw_data: bytes) -> bool:
        """Validate code is processable"""
        try:
            code = raw_data.decode('utf-8')
            # Try to compile (doesn't execute)
            compile(code, '<string>', 'exec')
            return True
        except:
            return False
    
    def _detect_language(self, code: str, metadata: Dict) -> str:
        """Detect programming language"""
        if 'def ' in code and 'import' in code:
            return "python"
        elif 'function' in code and ('const' in code or 'let' in code):
            return "javascript"
        elif 'func ' in code and 'package' in code:
            return "go"
        return metadata.get("language", "unknown")
    
    def _extract_functions(self, code: str) -> List[str]:
        """Extract function names"""
        import re
        # Python functions
        py_funcs = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        # JavaScript functions
        js_funcs = re.findall(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        return py_funcs + js_funcs
    
    def _extract_classes(self, code: str) -> List[str]:
        """Extract class names"""
        import re
        return re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements"""
        import re
        py_imports = re.findall(r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', code)
        return list(set(py_imports))


class DocumentAdapter(DataAdapter):
    """Adapter for documents (PDF, Word, Text, Markdown)"""
    
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process document data"""
        logger.info("DocumentAdapter: Processing document...")
        
        try:
            # Try to decode as text
            text = raw_data.decode('utf-8')
            
            # Extract metadata
            word_count = len(text.split())
            char_count = len(text)
            line_count = len(text.split('\n'))
            
            # Chunk for embeddings (1000 char chunks)
            chunks = self._create_chunks(text, chunk_size=1000)
            
            return {
                "success": True,
                "text": text,
                "word_count": word_count,
                "char_count": char_count,
                "line_count": line_count,
                "chunks": chunks,
                "language": "en"  # Would detect language
            }
        except Exception as e:
            logger.error(f"DocumentAdapter error: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, raw_data: bytes) -> bool:
        """Validate document is processable"""
        try:
            raw_data.decode('utf-8')
            return True
        except:
            return False
    
    def _create_chunks(self, text: str, chunk_size: int = 1000) -> List[Dict]:
        """Split text into chunks for processing"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            chunks.append({
                "index": len(chunks),
                "text": chunk_text,
                "length": len(chunk_text)
            })
        return chunks


class MediaAdapter(DataAdapter):
    """Adapter for media (images, audio, video)"""
    
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process media data"""
        logger.info("MediaAdapter: Processing media...")
        
        media_type = self._detect_media_type(raw_data, metadata)
        
        if media_type == "image":
            result = await self._process_image(raw_data)
        elif media_type == "audio":
            result = await self._process_audio(raw_data)
        elif media_type == "video":
            result = await self._process_video(raw_data)
        else:
            result = {"success": False, "error": "Unknown media type"}
        
        return {
            "media_type": media_type,
            **result
        }
    
    async def validate(self, raw_data: bytes) -> bool:
        """Validate media is processable"""
        # Check for common media file signatures
        signatures = {
            b'\xFF\xD8\xFF': 'jpeg',
            b'\x89PNG': 'png',
            b'GIF89a': 'gif',
            b'RIFF': 'wav',
        }
        
        for sig in signatures:
            if raw_data.startswith(sig):
                return True
        return False
    
    def _detect_media_type(self, raw_data: bytes, metadata: Dict) -> str:
        """Detect media type"""
        filename = metadata.get("filename", "")
        
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            return "image"
        elif filename.endswith(('.mp3', '.wav', '.ogg')):
            return "audio"
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            return "video"
        
        # Check file signatures
        if raw_data.startswith(b'\xFF\xD8') or raw_data.startswith(b'\x89PNG'):
            return "image"
        elif raw_data.startswith(b'RIFF'):
            return "audio"
        
        return "unknown"
    
    async def _process_image(self, raw_data: bytes) -> Dict:
        """Process image (OCR, analysis)"""
        # In production: use OCR, computer vision
        return {
            "success": True,
            "size_bytes": len(raw_data),
            "ocr_text": "[OCR would extract text here]",
            "detected_objects": []
        }
    
    async def _process_audio(self, raw_data: bytes) -> Dict:
        """Process audio (ASR, transcription)"""
        # In production: use Whisper or other ASR
        return {
            "success": True,
            "size_bytes": len(raw_data),
            "transcript": "[ASR would transcribe here]",
            "duration_seconds": 0
        }
    
    async def _process_video(self, raw_data: bytes) -> Dict:
        """Process video"""
        # In production: extract frames, audio, metadata
        return {
            "success": True,
            "size_bytes": len(raw_data),
            "frame_count": 0,
            "duration_seconds": 0
        }


class StructuredAdapter(DataAdapter):
    """Adapter for structured data (CSV, JSON, Parquet)"""
    
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process structured data"""
        logger.info("StructuredAdapter: Processing structured data...")
        
        try:
            # Try JSON first
            import json
            data = json.loads(raw_data.decode('utf-8'))
            format_type = "json"
            
            return {
                "success": True,
                "format": format_type,
                "data": data,
                "record_count": len(data) if isinstance(data, list) else 1,
                "schema": self._infer_schema(data)
            }
        except:
            # Try CSV
            try:
                text = raw_data.decode('utf-8')
                lines = text.split('\n')
                
                return {
                    "success": True,
                    "format": "csv",
                    "row_count": len(lines),
                    "columns": lines[0].split(',') if lines else []
                }
            except Exception as e:
                logger.error(f"StructuredAdapter error: {e}")
                return {"success": False, "error": str(e)}
    
    async def validate(self, raw_data: bytes) -> bool:
        """Validate structured data"""
        try:
            import json
            json.loads(raw_data.decode('utf-8'))
            return True
        except:
            try:
                text = raw_data.decode('utf-8')
                return ',' in text or '\t' in text  # CSV/TSV
            except:
                return False
    
    def _infer_schema(self, data: Any) -> Dict:
        """Infer schema from data"""
        if isinstance(data, dict):
            return {k: type(v).__name__ for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 0:
            return self._infer_schema(data[0])
        return {}


class WebAdapter(DataAdapter):
    """Adapter for web content (HTML, APIs, URLs)"""
    
    async def process(self, raw_data: bytes, metadata: Dict) -> Dict[str, Any]:
        """Process web content"""
        logger.info("WebAdapter: Processing web content...")
        
        try:
            html = raw_data.decode('utf-8')
            
            # Extract text content
            text = self._extract_text(html)
            
            # Extract links
            links = self._extract_links(html)
            
            return {
                "success": True,
                "html": html,
                "text": text,
                "links": links,
                "size_bytes": len(raw_data)
            }
        except Exception as e:
            logger.error(f"WebAdapter error: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate(self, raw_data: bytes) -> bool:
        """Validate web content"""
        try:
            content = raw_data.decode('utf-8')
            return '<html' in content.lower() or '<div' in content.lower()
        except:
            return False
    
    def _extract_text(self, html: str) -> str:
        """Extract text from HTML"""
        # Simple text extraction (in production: use BeautifulSoup)
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_links(self, html: str) -> List[str]:
        """Extract links from HTML"""
        import re
        return re.findall(r'href=["\']([^"\']+)["\']', html)


# Adapter registry
ADAPTERS = {
    "code": CodeAdapter(),
    "document": DocumentAdapter(),
    "media": MediaAdapter(),
    "structured": StructuredAdapter(),
    "web": WebAdapter()
}


def get_adapter(data_type: str) -> Optional[DataAdapter]:
    """Get adapter for data type"""
    return ADAPTERS.get(data_type)

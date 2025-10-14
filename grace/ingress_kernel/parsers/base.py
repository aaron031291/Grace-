"""
Content Parsers - Transform raw content into structured data.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime

from grace.contracts.ingress_contracts import RawEvent


logger = logging.getLogger(__name__)


class ParseResult:
    """Result of parsing operation."""

    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
        bytes_in: int = 0,
        bytes_out: int = 0,
    ):
        self.success = success
        self.data = data or {}
        self.errors = errors or []
        self.bytes_in = bytes_in
        self.bytes_out = bytes_out


class BaseParser(ABC):
    """Base class for content parsers."""

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = options or {}

    @abstractmethod
    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse raw event content."""
        pass

    def _calculate_bytes(self, content: Any) -> int:
        """Calculate byte size of content."""
        if isinstance(content, bytes):
            return len(content)
        elif isinstance(content, str):
            return len(content.encode("utf-8"))
        else:
            return len(json.dumps(content).encode("utf-8"))


class JSONParser(BaseParser):
    """JSON content parser."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse JSON content."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            if isinstance(raw_event.payload, dict):
                # Already parsed
                data = raw_event.payload
            elif isinstance(raw_event.payload, str):
                data = json.loads(raw_event.payload)
            elif isinstance(raw_event.payload, bytes):
                data = json.loads(raw_event.payload.decode("utf-8"))
            else:
                raise ValueError(f"Unsupported payload type: {type(raw_event.payload)}")

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                errors=[f"JSON decode error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )
        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"JSON parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )


class CSVParser(BaseParser):
    """CSV content parser."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse CSV content."""
        try:
            import csv
            from io import StringIO

            bytes_in = self._calculate_bytes(raw_event.payload)

            # Convert to string if needed
            if isinstance(raw_event.payload, bytes):
                content = raw_event.payload.decode("utf-8")
            else:
                content = str(raw_event.payload)

            # Parse CSV
            csv_reader = csv.DictReader(StringIO(content))
            rows = list(csv_reader)

            data = {
                "format": "csv",
                "headers": csv_reader.fieldnames or [],
                "rows": rows,
                "row_count": len(rows),
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"CSV parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )


class HTMLParser(BaseParser):
    """HTML content parser."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse HTML content."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            # Convert to string if needed
            if isinstance(raw_event.payload, bytes):
                content = raw_event.payload.decode("utf-8")
            else:
                content = str(raw_event.payload)

            # Mock HTML parsing - would use BeautifulSoup or similar
            # Extract basic elements
            data = {
                "format": "html",
                "title": self._extract_title(content),
                "text": self._extract_text(content),
                "links": self._extract_links(content),
                "meta": self._extract_meta(content),
                "raw_html": content[:1000]
                if len(content) > 1000
                else content,  # Truncate
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"HTML parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        # Simple regex-based extraction (mock)
        import re

        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else "No title"

    def _extract_text(self, html: str) -> str:
        """Extract text content from HTML."""
        # Mock text extraction - would remove HTML tags properly
        import re

        text = re.sub(r"<[^>]+>", "", html)
        return " ".join(text.split())[:500]  # Truncate

    def _extract_links(self, html: str) -> List[str]:
        """Extract links from HTML."""
        # Mock link extraction
        import re

        links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
        return links[:10]  # Limit to first 10 links

    def _extract_meta(self, html: str) -> Dict[str, str]:
        """Extract meta information from HTML."""
        # Mock meta extraction
        return {
            "charset": "utf-8",
            "language": "en",
            "parsed_at": datetime.utcnow().isoformat(),
        }


class PDFParser(BaseParser):
    """PDF content parser."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse PDF content."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            # Mock PDF parsing - would use PyPDF2 or similar
            data = {
                "format": "pdf",
                "text": "Mock PDF text content extracted from binary data",
                "pages": 1,
                "metadata": {
                    "title": "Sample PDF Document",
                    "author": "Unknown",
                    "creation_date": datetime.utcnow().isoformat(),
                },
                "extraction_method": "mock",
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"PDF parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )


class AudioParser(BaseParser):
    """Audio content parser (ASR - Automatic Speech Recognition)."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse audio content using ASR."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            # Mock ASR processing - would use Whisper or similar
            data = {
                "format": "audio",
                "transcript": "This is a mock transcript of the audio content",
                "language": "en",
                "confidence": 0.92,
                "duration_seconds": 120.5,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 5.2,
                        "text": "This is a mock transcript",
                        "confidence": 0.95,
                    },
                    {
                        "start": 5.2,
                        "end": 10.8,
                        "text": "of the audio content",
                        "confidence": 0.89,
                    },
                ],
                "processing_method": "mock_asr",
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"Audio parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )


class VideoParser(BaseParser):
    """Video content parser (generates VTT subtitles)."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse video content to extract audio and generate VTT."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            # Mock video processing - would extract audio then run ASR
            data = {
                "format": "video",
                "transcript": "Mock transcript from video audio track",
                "vtt": "WEBVTT\\n\\n00:00:00.000 --> 00:00:05.200\\nMock transcript from video\\n\\n00:00:05.200 --> 00:00:10.800\\naudio track",
                "language": "en",
                "duration_seconds": 240.0,
                "video_metadata": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 30,
                    "codec": "h264",
                },
                "audio_metadata": {"sample_rate": 44100, "channels": 2, "codec": "aac"},
                "processing_method": "mock_video_asr",
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"Video parsing error: {str(e)}"],
                bytes_in=bytes_in,
            )


class XMLParser(BaseParser):
    """XML content parser."""

    async def parse(self, raw_event: RawEvent) -> ParseResult:
        """Parse XML content."""
        try:
            bytes_in = self._calculate_bytes(raw_event.payload)

            # Convert to string if needed
            if isinstance(raw_event.payload, bytes):
                content = raw_event.payload.decode("utf-8")
            else:
                content = str(raw_event.payload)

            # Mock XML parsing - would use lxml or ElementTree
            data = {
                "format": "xml",
                "root_element": "document",
                "namespace": None,
                "elements": [
                    {"tag": "title", "text": "Sample XML Document"},
                    {"tag": "content", "text": "Mock XML content"},
                ],
                "attributes": {},
                "raw_xml": content[:500] if len(content) > 500 else content,
            }

            bytes_out = self._calculate_bytes(data)

            return ParseResult(
                success=True, data=data, bytes_in=bytes_in, bytes_out=bytes_out
            )

        except Exception as e:
            return ParseResult(
                success=False,
                errors=[f"XML parsing error: {str(e)}"],
                bytes_in=self._calculate_bytes(raw_event.payload),
            )


class ParserFactory:
    """Factory for creating content parsers."""

    _parsers = {
        "json": JSONParser,
        "csv": CSVParser,
        "html": HTMLParser,
        "pdf": PDFParser,
        "audio": AudioParser,
        "video": VideoParser,
        "xml": XMLParser,
    }

    @classmethod
    def create_parser(
        cls, parser_type: str, options: Optional[Dict[str, Any]] = None
    ) -> BaseParser:
        """Create parser based on type."""
        parser_class = cls._parsers.get(parser_type.lower())

        if not parser_class:
            raise ValueError(f"Unsupported parser type: {parser_type}")

        return parser_class(options)

    @classmethod
    def register_parser(cls, parser_type: str, parser_class: type):
        """Register a new parser type."""
        cls._parsers[parser_type.lower()] = parser_class

    @classmethod
    def list_supported_types(cls) -> List[str]:
        """List supported parser types."""
        return list(cls._parsers.keys())

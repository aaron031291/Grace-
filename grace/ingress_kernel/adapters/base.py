"""
Source Adapters - Connectors for different data sources.
"""
import asyncio
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from grace.utils.time import now_utc, iso_now_utc
from typing import Any, Dict, Optional, AsyncIterator
import urllib.parse
import uuid

from grace.contracts.ingress_contracts import SourceConfig, RawEvent, generate_event_id


logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Base class for all source adapters."""
    
    def __init__(self, source_config: SourceConfig):
        self.source_config = source_config
        self.running = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the source."""
        pass
    
    @abstractmethod
    async def fetch(self) -> AsyncIterator[RawEvent]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to the source."""
        pass
    
    async def start(self):
        """Start the adapter."""
        if self.running:
            return
        
        self.running = True
        await self.connect()
        logger.info(f"Started adapter for {self.source_config.source_id}")
    
    async def stop(self):
        """Stop the adapter."""
        if not self.running:
            return
        
        self.running = False
        await self.disconnect()
        logger.info(f"Stopped adapter for {self.source_config.source_id}")
    
    def _create_raw_event(self, payload: Any, headers: Optional[Dict] = None, 
                         offset: Optional[str] = None) -> RawEvent:
        """Create a raw event from payload."""
        content_hash = self._compute_hash(payload)
        
        return RawEvent(
            event_id=generate_event_id(),
            source_id=self.source_config.source_id,
            kind=self.source_config.parser.value,
            payload=payload,
            headers=headers,
            offset=offset or f"auto_{iso_now_utc()}",
            hash=content_hash
        )
    
    def _compute_hash(self, payload: Any) -> str:
        """Compute hash for deduplication."""
        if isinstance(payload, bytes):
            return hashlib.sha256(payload).hexdigest()
        elif isinstance(payload, str):
            return hashlib.sha256(payload.encode()).hexdigest()
        else:
            content = json.dumps(payload, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()


class HTTPAdapter(BaseAdapter):
    """HTTP/HTTPS source adapter."""
    
    async def connect(self) -> bool:
        """Test HTTP connection."""
        try:
            # Mock connection test
            logger.info(f"Testing HTTP connection to {self.source_config.uri}")
            return True
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return False
    
    async def fetch(self) -> AsyncIterator[RawEvent]:
        """Fetch data from HTTP endpoint."""
        try:
            # Mock HTTP fetch - in real implementation would use aiohttp
            mock_data = {
                "title": "Sample HTTP Content",
                "content": f"Content from {self.source_config.uri}",
                "timestamp": iso_now_utc()
            }
            
            yield self._create_raw_event(
                payload=mock_data,
                headers={"content-type": "application/json"},
                offset=f"http_{iso_now_utc()}"
            )
            
        except Exception as e:
            logger.error(f"HTTP fetch failed: {e}")
    
    async def disconnect(self):
        """Clean up HTTP resources."""
        pass


class RSSAdapter(BaseAdapter):
    """RSS feed adapter."""
    
    def __init__(self, source_config: SourceConfig):
        super().__init__(source_config)
        self.last_check = None
        self.seen_guids = set()
    
    async def connect(self) -> bool:
        """Test RSS feed connection."""
        try:
            logger.info(f"Testing RSS connection to {self.source_config.uri}")
            return True
        except Exception as e:
            logger.error(f"RSS connection failed: {e}")
            return False
    
    async def fetch(self) -> AsyncIterator[RawEvent]:
        """Fetch RSS items."""
        try:
            # Mock RSS data - in real implementation would parse XML
            mock_items = [
                {
                    "title": "Sample RSS Item 1",
                    "description": "First RSS item description",
                    "link": f"{self.source_config.uri}/item1",
                    "pubDate": iso_now_utc(),
                    "guid": str(uuid.uuid4())
                },
                {
                    "title": "Sample RSS Item 2", 
                    "description": "Second RSS item description",
                    "link": f"{self.source_config.uri}/item2",
                    "pubDate": iso_now_utc(),
                    "guid": str(uuid.uuid4())
                }
            ]
            
            for item in mock_items:
                if item["guid"] not in self.seen_guids:
                    self.seen_guids.add(item["guid"])
                    yield self._create_raw_event(
                        payload=item,
                        offset=f"rss_{item['guid']}"
                    )
            
        except Exception as e:
            logger.error(f"RSS fetch failed: {e}")
    
    async def disconnect(self):
        """Clean up RSS resources."""
        self.seen_guids.clear()


class S3Adapter(BaseAdapter):
    """Amazon S3 adapter."""
    
    def __init__(self, source_config: SourceConfig):
        super().__init__(source_config)
        self.bucket = None
        self.prefix = None
        self._parse_s3_uri()
    
    def _parse_s3_uri(self):
        """Parse S3 URI into bucket and prefix."""
        if self.source_config.uri.startswith("s3://"):
            uri_parts = self.source_config.uri[5:].split("/", 1)
            self.bucket = uri_parts[0]
            self.prefix = uri_parts[1] if len(uri_parts) > 1 else ""
    
    async def connect(self) -> bool:
        """Test S3 connection."""
        try:
            logger.info(f"Testing S3 connection to bucket {self.bucket}")
            # Mock S3 connection - would use boto3
            return True
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            return False
    
    async def fetch(self) -> AsyncIterator[RawEvent]:
        """Fetch objects from S3."""
        try:
            # Mock S3 objects
            mock_objects = [
                {
                    "key": f"{self.prefix}/file1.json",
                    "content": {"data": "S3 file 1 content"},
                    "last_modified": iso_now_utc()
                },
                {
                    "key": f"{self.prefix}/file2.json",
                    "content": {"data": "S3 file 2 content"},
                        "last_modified": iso_now_utc()
                }
            ]
            
            for obj in mock_objects:
                yield self._create_raw_event(
                    payload=obj["content"],
                    headers={
                        "s3-bucket": self.bucket,
                        "s3-key": obj["key"],
                        "last-modified": obj["last_modified"]
                    },
                    offset=f"s3_{obj['key']}"
                )
            
        except Exception as e:
            logger.error(f"S3 fetch failed: {e}")
    
    async def disconnect(self):
        """Clean up S3 resources."""
        pass


class GitHubAdapter(BaseAdapter):
    """GitHub repository adapter."""
    
    def __init__(self, source_config: SourceConfig):
        super().__init__(source_config)
        self.repo_owner = None
        self.repo_name = None
        self._parse_github_uri()
    
    def _parse_github_uri(self):
        """Parse GitHub URI."""
        if "github.com" in self.source_config.uri:
            path_parts = urllib.parse.urlparse(self.source_config.uri).path.strip("/").split("/")
            if len(path_parts) >= 2:
                self.repo_owner = path_parts[0]
                self.repo_name = path_parts[1]
    
    async def connect(self) -> bool:
        """Test GitHub API connection."""
        try:
            logger.info(f"Testing GitHub connection to {self.repo_owner}/{self.repo_name}")
            return True
        except Exception as e:
            logger.error(f"GitHub connection failed: {e}")
            return False
    
    async def fetch(self) -> AsyncIterator[RawEvent]:
        """Fetch data from GitHub."""
        try:
            # Mock GitHub data - issues, PRs, commits, etc.
            mock_items = [
                {
                    "type": "issue",
                    "number": 1,
                    "title": "Sample Issue",
                    "body": "This is a sample GitHub issue",
                    "state": "open",
                    "created_at": iso_now_utc()
                },
                {
                    "type": "pull_request",
                    "number": 2,
                    "title": "Sample PR",
                    "body": "This is a sample pull request",
                    "state": "open",
                    "created_at": iso_now_utc()
                }
            ]
            
            for item in mock_items:
                yield self._create_raw_event(
                    payload=item,
                    headers={
                        "github-repo": f"{self.repo_owner}/{self.repo_name}",
                        "github-type": item["type"]
                    },
                    offset=f"github_{item['type']}_{item['number']}"
                )
            
        except Exception as e:
            logger.error(f"GitHub fetch failed: {e}")
    
    async def disconnect(self):
        """Clean up GitHub resources."""
        pass


class AdapterFactory:
    """Factory for creating source adapters."""
    
    _adapters = {
        "http": HTTPAdapter,
        "rss": RSSAdapter,
        "s3": S3Adapter,
        "github": GitHubAdapter
    }
    
    @classmethod
    def create_adapter(cls, source_config: SourceConfig) -> BaseAdapter:
        """Create adapter based on source configuration."""
        adapter_class = cls._adapters.get(source_config.kind)
        
        if not adapter_class:
            raise ValueError(f"Unsupported source kind: {source_config.kind}")
        
        return adapter_class(source_config)
    
    @classmethod
    def register_adapter(cls, kind: str, adapter_class: type):
        """Register a new adapter type."""
        cls._adapters[kind] = adapter_class
    
    @classmethod
    def list_supported_kinds(cls) -> list:
        """List supported source kinds."""
        return list(cls._adapters.keys())
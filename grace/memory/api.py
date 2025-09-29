"""
Grace Memory API Service - REST endpoints for memory operations.

Provides:
- /memory/write - Store content with processing
- /memory/read - Retrieve content by ID
- /memory/search - Search and rank content
- /memory/stats - Get memory system statistics
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None
    JSONResponse = None

from .lightning import LightningMemory
from .fusion import FusionMemory
from .librarian import EnhancedLibrarian
from ..core.utils import create_error_response, validate_request_size, utc_timestamp, normalize_timestamp
from ..core.middleware import get_request_id, RequestIDMiddleware, GraceHTTPExceptionHandler

logger = logging.getLogger(__name__)


# API Models
class MemoryWriteRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request to write content to memory."""
    content: str = Field(..., description="Content to store")
    source_id: Optional[str] = Field(None, description="Optional source identifier")
    content_type: str = Field(default="text/plain", description="Content MIME type")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


class MemoryReadRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request to read content from memory."""
    key: str = Field(..., description="Memory key to retrieve")
    verify_integrity: bool = Field(default=True, description="Verify content integrity")


class MemorySearchRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request to search memory."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    min_constitutional_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum constitutional score")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")


class MemoryResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Memory operation response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class GraceMemoryAPI:
    """
    Grace Memory API Service providing REST endpoints for memory operations.
    
    Integrates Lightning (cache), Fusion (storage), and Librarian (processing).
    """
    
    def __init__(self,
                 lightning_memory: Optional[LightningMemory] = None,
                 fusion_memory: Optional[FusionMemory] = None,
                 librarian: Optional[EnhancedLibrarian] = None):
        
        # Initialize memory components
        self.lightning = lightning_memory or LightningMemory(
            max_size=10000,
            default_ttl=3600
        )
        
        self.fusion = fusion_memory or FusionMemory(
            storage_path="/tmp/grace_fusion_memory",
            enable_compression=True
        )
        
        self.librarian = librarian or EnhancedLibrarian(
            lightning_memory=self.lightning,
            fusion_memory=self.fusion
        )
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Grace Memory API",
                description="Memory operations for Grace kernel",
                version="1.0.0"
            )
            
            # Add middleware
            self.app.add_middleware(RequestIDMiddleware)
            
            # Add exception handlers
            @self.app.exception_handler(HTTPException)
            async def http_exception_handler(request: Request, exc: HTTPException):
                return await GraceHTTPExceptionHandler.handle_http_exception(request, exc)

            @self.app.exception_handler(Exception)
            async def general_exception_handler(request: Request, exc: Exception):
                return await GraceHTTPExceptionHandler.handle_http_exception(request, exc)
            
            self._register_routes()
        else:
            logger.warning("FastAPI not available, running in compatibility mode")
            self.app = None
    
    def _register_routes(self):
        """Register FastAPI routes."""
        if not self.app:
            return
        
        @self.app.post("/api/memory/v1/write", response_model=MemoryResponse)
        async def write_memory(request: MemoryWriteRequest):
            """Write content to memory with full processing pipeline."""
            try:
                # Validate content size
                size_error = validate_request_size(request.content, max_size_mb=10)
                if size_error:
                    return JSONResponse(status_code=413, content=size_error)
                
                # Process through librarian for chunking and indexing
                if request.content_type == "text/plain" or request.content_type.startswith("text/"):
                    result = self.librarian.ingest_document(
                        content=request.content,
                        source_id=request.source_id,
                        metadata={
                            "content_type": request.content_type,
                            "tags": request.tags,
                            "cache_ttl": request.cache_ttl,
                            **request.metadata
                        }
                    )
                    
                    if result["status"] == "success":
                        return MemoryResponse(
                            success=True,
                            message=f"Content processed: {result['chunks_processed']} chunks created",
                            data={
                                "source_id": result["source_id"],
                                "document_entry_id": result["document_entry_id"],
                                "chunks_processed": result["chunks_processed"],
                                "constitutional_score": result["constitutional_score"],
                                "timestamp": utc_timestamp(),
                                "request_id": get_request_id()
                            }
                        )
                    else:
                        return JSONResponse(
                            status_code=400,
                            content=create_error_response(
                                "MEMORY_PROCESSING_FAILED",
                                f"Processing failed: {result.get('reason', 'Unknown error')}",
                                str(result)
                            )
                        )
                
                else:
                    # Direct storage for non-text content
                    cache_key = request.source_id or f"direct_{utc_timestamp().replace(':', '').replace('-', '').replace('.', '')[:16]}"
                    
                    # Store in cache
                    cache_success = self.lightning.put(
                        cache_key,
                        request.content,
                        ttl_seconds=request.cache_ttl,
                        tags=request.tags
                    )
                    
                    # Store in long-term storage
                    fusion_entry_id = self.fusion.write(
                        key=cache_key,
                        value=request.content,
                        content_type=request.content_type,
                        tags=request.tags,
                        metadata=request.metadata
                    )
                    
                    return MemoryResponse(
                        success=cache_success and fusion_entry_id is not None,
                        message="Content stored directly",
                        data={
                            "cache_key": cache_key,
                            "fusion_entry_id": fusion_entry_id,
                            "cache_stored": cache_success,
                            "timestamp": utc_timestamp(),
                            "request_id": get_request_id()
                        }
                    )
                    
            except Exception as e:
                logger.error(f"Memory write failed: {e}", extra={"request_id": get_request_id()})
                return JSONResponse(
                    status_code=500,
                    content=create_error_response("MEMORY_WRITE_FAILED", "Failed to write to memory", str(e))
                )
        
        @self.app.post("/api/memory/v1/read", response_model=MemoryResponse)
        async def read_memory(request: MemoryReadRequest):
            """Read content from memory."""
            try:
                # Try Lightning cache first
                content = self.lightning.get(request.key, verify_integrity=request.verify_integrity)
                
                if content is not None:
                    return MemoryResponse(
                        success=True,
                        message="Content retrieved from cache",
                        data={
                            "key": request.key,
                            "content": content,
                            "source": "cache"
                        }
                    )
                
                # Try Fusion storage
                fusion_entries = self.fusion.search(key_pattern=request.key, limit=1)
                if fusion_entries:
                    entry = fusion_entries[0]
                    
                    # Cache for future access
                    self.lightning.put(request.key, entry.value, ttl_seconds=1800)
                    
                    return MemoryResponse(
                        success=True,
                        message="Content retrieved from storage",
                        data={
                            "key": request.key,
                            "content": entry.value,
                            "source": "storage",
                            "entry_id": entry.entry_id,
                            "created_at": entry.created_at.isoformat()
                        }
                    )
                
                return MemoryResponse(
                    success=False,
                    message=f"Content not found for key: {request.key}"
                )
                
            except Exception as e:
                logger.error(f"Memory read failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/memory/v1/search", response_model=MemoryResponse)
        async def search_memory(request: MemorySearchRequest):
            """Search memory content."""
            try:
                # Use librarian for intelligent search
                results = self.librarian.search_and_rank(
                    query=request.query,
                    limit=request.limit,
                    min_constitutional_score=request.min_constitutional_score
                )
                
                # Apply additional filters if specified
                if request.content_type or request.tags:
                    filtered_results = []
                    for result in results:
                        metadata = result.get("metadata", {})
                        
                        # Content type filter
                        if request.content_type:
                            if metadata.get("content_type") != request.content_type:
                                continue
                        
                        # Tags filter
                        if request.tags:
                            result_tags = metadata.get("tags", [])
                            if not any(tag in result_tags for tag in request.tags):
                                continue
                        
                        filtered_results.append(result)
                    
                    results = filtered_results
                
                return MemoryResponse(
                    success=True,
                    message=f"Search completed: {len(results)} results found",
                    data={
                        "query": request.query,
                        "results_count": len(results),
                        "results": results
                    }
                )
                
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/memory/v1/stats", response_model=MemoryResponse)
        async def get_memory_stats():
            """Get memory system statistics."""
            try:
                lightning_stats = self.lightning.get_stats()
                fusion_stats = self.fusion.get_stats()
                librarian_stats = self.librarian.get_stats()
                
                # Perform health checks
                lightning_health = self.lightning.health_check()
                
                return MemoryResponse(
                    success=True,
                    message="Memory statistics retrieved",
                    data={
                        "timestamp": utc_timestamp(),
                        "request_id": get_request_id(),
                        "lightning_cache": {
                            "stats": lightning_stats,
                            "health": lightning_health
                        },
                        "fusion_storage": fusion_stats,
                        "librarian": librarian_stats,
                        "overall_health": {
                            "healthy": lightning_health["healthy"],
                            "total_memory_entries": lightning_stats["entries"] + fusion_stats.get("active_entries", 0)
                        }
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}", extra={"request_id": get_request_id()})
                return JSONResponse(
                    status_code=500,
                    content=create_error_response("MEMORY_STATS_FAILED", "Failed to get memory statistics", str(e))
                )
        
        @self.app.post("/api/memory/v1/document/{source_id}", response_model=MemoryResponse)
        async def get_document_info(source_id: str):
            """Get information about a specific document."""
            try:
                doc_info = self.librarian.get_document_info(source_id)
                
                if doc_info:
                    return MemoryResponse(
                        success=True,
                        message="Document information retrieved",
                        data=doc_info
                    )
                else:
                    return MemoryResponse(
                        success=False,
                        message=f"Document not found: {source_id}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to get document info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/memory/v1/cache/{key}")
        async def clear_cache_key(key: str):
            """Clear specific key from cache."""
            try:
                success = self.lightning.delete(key)
                
                return MemoryResponse(
                    success=success,
                    message=f"Cache key {'cleared' if success else 'not found'}: {key}"
                )
                
            except Exception as e:
                logger.error(f"Failed to clear cache key: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/memory/v1/integrity/verify")
        async def verify_integrity():
            """Verify memory system integrity."""
            try:
                # Verify Fusion storage integrity
                fusion_integrity = self.fusion.verify_integrity()
                
                # Verify Lightning cache health
                cache_health = self.lightning.health_check()
                
                overall_healthy = (
                    fusion_integrity.get("corrupted_entries", 0) == 0 and
                    cache_health["healthy"]
                )
                
                return MemoryResponse(
                    success=True,
                    message="Integrity verification completed",
                    data={
                        "overall_healthy": overall_healthy,
                        "fusion_integrity": fusion_integrity,
                        "cache_health": cache_health,
                        "verified_at": datetime.utcnow().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Integrity verification failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # Standalone methods for non-FastAPI usage
    async def write_content(self, 
                          content: str, 
                          source_id: str = None,
                          content_type: str = "text/plain",
                          tags: List[str] = None,
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Write content to memory (standalone method)."""
        result = self.librarian.ingest_document(
            content=content,
            source_id=source_id,
            metadata={
                "content_type": content_type,
                "tags": tags or [],
                **(metadata or {})
            }
        )
        return result
    
    async def read_content(self, key: str) -> Optional[Any]:
        """Read content from memory (standalone method)."""
        # Try cache first
        content = self.lightning.get(key)
        if content is not None:
            return content
        
        # Try storage
        entries = self.fusion.search(key_pattern=key, limit=1)
        if entries:
            return entries[0].value
        
        return None
    
    async def search_content(self, 
                           query: str, 
                           limit: int = 10,
                           min_constitutional_score: float = 0.7) -> List[Dict[str, Any]]:
        """Search content (standalone method)."""
        return self.librarian.search_and_rank(
            query=query,
            limit=limit,
            min_constitutional_score=min_constitutional_score
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics (standalone method)."""
        return {
            "lightning": self.lightning.get_stats(),
            "fusion": self.fusion.get_stats(),
            "librarian": self.librarian.get_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Compatibility wrapper for environments without FastAPI
if not FASTAPI_AVAILABLE:
    logger.warning("FastAPI not available. Memory API running in compatibility mode.")
    
    class MockMemoryAPI:
        """Mock API for environments without FastAPI."""
        
        def __init__(self, **kwargs):
            self.app = None
            self.lightning = LightningMemory()
            self.fusion = FusionMemory()
            self.librarian = EnhancedLibrarian(self.lightning, self.fusion)
            logger.info("Mock Memory API initialized")
        
        async def write_content(self, content: str, **kwargs):
            return await self.librarian.ingest_document(content)
        
        async def read_content(self, key: str):
            return self.lightning.get(key)
        
        async def search_content(self, query: str, **kwargs):
            return self.librarian.search_and_rank(query)
        
        def get_stats(self):
            return {"status": "mock", "message": "FastAPI not available"}
    
    GraceMemoryAPI = MockMemoryAPI
"""
Patterns MCP Handler

Canonical API for pattern storage, retrieval, and semantic search.
Patterns are observations of system behaviors, user intents, or recurring events.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from grace.mcp.base_mcp import BaseMCP, mcp_endpoint, MCPContext
from grace.mcp.pushback import handle_index_failure, retry_with_backoff


# --- Request/Response Schemas ---

class PatternCreateRequest(BaseModel):
    """Request to create a new pattern"""
    who: str = Field(..., description="Who observed/created this pattern")
    what: str = Field(..., description="What is the pattern about")
    where: str = Field(..., description="Where was it observed (component/domain)")
    when: float = Field(..., description="When was it observed (timestamp)")
    why: Optional[str] = Field(None, description="Why is this pattern significant")
    how: Optional[str] = Field(None, description="How was it detected")
    raw_text: str = Field(..., description="Full text description for embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "who": "intelligence_kernel",
                "what": "Model latency increased",
                "where": "sentiment_model_v2.3",
                "when": 1697270400.0,
                "why": "Data drift detected",
                "how": "Automatic monitoring",
                "raw_text": "Intelligence kernel detected latency increase in sentiment model v2.3 due to data drift",
                "metadata": {"model_id": "sentiment_v2.3", "latency_ms": 87},
                "tags": ["performance", "data_drift", "ml"]
            }
        }
    )


class PatternResponse(BaseModel):
    """Response after creating a pattern"""
    id: str
    observation_id: str
    vector_id: Optional[str] = None
    trust_score: float
    created_at: float


class SemanticSearchRequest(BaseModel):
    """Request for semantic pattern search"""
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    trust_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum trust score")
    include_metadata: bool = Field(True, description="Include full metadata in results")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "model performance degradation patterns",
                "top_k": 5,
                "filters": {"tags": "performance"},
                "trust_threshold": 0.7
            }
        }
    )


class PatternSearchResult(BaseModel):
    """Single search result"""
    id: str
    who: str
    what: str
    where: str
    when: float
    similarity: float
    trust_score: float
    metadata: Optional[Dict[str, Any]] = None


class SemanticSearchResponse(BaseModel):
    """Response from semantic search"""
    query: str
    results: List[PatternSearchResult]
    total_found: int
    search_time_ms: float


class PatternMetadataRequest(BaseModel):
    """Request for pattern metadata/provenance"""
    pattern_id: str


class PatternMetadataResponse(BaseModel):
    """Pattern metadata and provenance"""
    id: str
    trust_score: float
    provenance: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    related_observations: List[str]
    related_evaluations: List[str]


# --- MCP Handler ---

class PatternsMCP(BaseMCP):
    """
    MCP handler for patterns domain.
    
    Provides:
    - Pattern creation with automatic vectorization
    - Semantic search across patterns
    - Metadata/provenance retrieval
    - Trust scoring and lineage tracking
    """
    
    domain = "patterns"
    manifest_path = "grace/mcp/manifests/patterns.yaml"
    collection_name = "patterns_vectors"
    
    @mcp_endpoint(manifest="patterns.yaml", endpoint="create")
    async def create_pattern(self, 
                            request: PatternCreateRequest, 
                            context: MCPContext) -> PatternResponse:
        """
        Create a new pattern with automatic vectorization.
        
        Flow:
        1. Validate request (MCP framework)
        2. Governance check (MCP framework)
        3. Persist to database
        4. Embed text with retry
        5. Upsert to vector store
        6. Return response
        
        Framework automatically:
        - Records observation (O-Loop)
        - Emits PATTERN.CREATED event
        - Logs to immutable audit
        - Evaluates outcome (E-Loop)
        """
        import time
        
        # 1. Prepare record for database
        pattern_data = {
            "who": request.who,
            "what": request.what,
            "where": request.where,
            "when": request.when,
            "why": request.why,
            "how": request.how,
            "raw_text": request.raw_text,
            "metadata": request.metadata,
            "tags": request.tags,
            "trust_score": context.caller.trust_score,
            "created_by": context.caller.id,
            "created_at": time.time()
        }
        
        # 2. Persist to database (observations table or dedicated patterns table)
        record_id = await self.db.insert('observations', {
            "observation_type": "pattern",
            "source_module": request.who,
            "observation_data": pattern_data,
            "context": {
                "created_via": "mcp",
                "caller": context.caller.id
            },
            "credibility_score": context.caller.trust_score,
            "observed_at": time.time()
        })
        
        # 3. Vectorize with retry (handles embedding failures)
        try:
            embedding = await retry_with_backoff(
                self.vectorize,
                max_attempts=3,
                initial_delay=1.0,
                text=request.raw_text,
                metadata={"id": record_id, "domain": "patterns"}
            )
        except Exception as e:
            # Framework will handle pushback, but we can add context
            await handle_index_failure(
                domain=self.domain,
                record_id=record_id,
                error=e,
                caller_id=context.caller.id,
                request_id=context.request_id,
                attempt=3
            )
            # Return response without vector_id (degraded mode)
            return PatternResponse(
                id=record_id,
                observation_id=record_id,
                vector_id=None,
                trust_score=context.caller.trust_score,
                created_at=time.time()
            )
        
        # 4. Upsert to vector store
        vector_metadata = {
            "id": record_id,
            "who": request.who,
            "where": request.where,
            "when": request.when,
            "tags": request.tags,
            "trust_score": context.caller.trust_score
        }
        
        try:
            await self.upsert_vector(
                collection=self.collection_name,
                id=record_id,
                embedding=embedding,
                metadata=vector_metadata
            )
        except Exception as e:
            # Log failure but don't fail request (degraded mode)
            await handle_index_failure(
                domain=self.domain,
                record_id=record_id,
                error=e,
                caller_id=context.caller.id,
                request_id=context.request_id,
                attempt=1
            )
        
        # 5. Return response (framework wraps in MCPResponse)
        return PatternResponse(
            id=record_id,
            observation_id=record_id,
            vector_id=record_id,
            trust_score=context.caller.trust_score,
            created_at=time.time()
        )
    
    @mcp_endpoint(manifest="patterns.yaml", endpoint="semantic_search")
    async def semantic_search(self,
                             request: SemanticSearchRequest,
                             context: MCPContext) -> SemanticSearchResponse:
        """
        Perform semantic search across patterns.
        
        Uses:
        - Lightning cache for hot queries
        - Vector similarity search
        - Trust-based filtering
        - Fusion query for full records
        """
        import time
        start_time = time.time()
        
        # Perform semantic search (handles caching internally)
        # Call the parent class semantic_search method with base parameters
        results = await super().semantic_search(
            collection=self.collection_name,
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            trust_threshold=request.trust_threshold
        )
        
        # Transform to response format
        search_results = []
        for r in results:
            data = r.get('observation_data', {})
            search_results.append(PatternSearchResult(
                id=r['observation_id'],
                who=data.get('who', 'unknown'),
                what=data.get('what', ''),
                where=data.get('where', ''),
                when=data.get('when', 0.0),
                similarity=r.get('similarity', 0.0),
                trust_score=r.get('credibility_score', 0.0),
                metadata=data.get('metadata') if request.include_metadata else None
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SemanticSearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_time_ms=elapsed_ms
        )
    
    @mcp_endpoint(manifest="patterns.yaml", endpoint="get_metadata")
    async def get_metadata(self,
                          request: PatternMetadataRequest,
                          context: MCPContext) -> PatternMetadataResponse:
        """
        Get pattern metadata, provenance, and lineage.
        
        Returns:
        - Trust score history
        - Audit trail
        - Related observations
        - Related evaluations (if pattern led to actions)
        """
        # Fetch pattern
        pattern = await self.db.query_one(
            "SELECT * FROM observations WHERE observation_id = ?",
            (request.pattern_id,)
        )
        
        if not pattern:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Pattern not found")
        
        # Build provenance
        provenance = {
            "created_by": pattern.get('source_module'),
            "created_at": pattern.get('observed_at'),
            "credibility_score": pattern.get('credibility_score'),
            "source": "mcp.patterns"
        }
        
        # Get audit trail
        audit_trail = await self.db.query_many("""
            SELECT action, timestamp, caller_id, severity
            FROM audit_logs
            WHERE json_extract(payload, '$.pattern_id') = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (request.pattern_id,))
        
        # Get related observations (if this pattern triggered others)
        related_obs = await self.db.query_many("""
            SELECT observation_id
            FROM observations
            WHERE json_extract(context, '$.triggered_by') = ?
            LIMIT 20
        """, (request.pattern_id,))
        
        # Get related evaluations (if actions were taken based on this pattern)
        related_evals = await self.db.query_many("""
            SELECT evaluation_id
            FROM evaluations
            WHERE json_extract(intended_outcome, '$.pattern_id') = ?
            LIMIT 20
        """, (request.pattern_id,))
        
        return PatternMetadataResponse(
            id=request.pattern_id,
            trust_score=pattern.get('credibility_score', 0.0),
            provenance=provenance,
            audit_trail=[dict(a) for a in audit_trail],
            related_observations=[r['observation_id'] for r in related_obs],
            related_evaluations=[r['evaluation_id'] for r in related_evals]
        )
    
    def _get_endpoint_config(self, endpoint_name: str) -> Dict[str, Any]:
        """Helper to get endpoint configuration from manifest"""
        for ep in self.manifest.get('endpoints', []):
            if ep['name'] == endpoint_name:
                return ep
        return {}


# --- FastAPI Integration Example ---

def register_patterns_mcp(app):
    """Register Patterns MCP with FastAPI app"""
    from fastapi import APIRouter, Depends, Request
    from grace.mcp.gateway import authenticate_caller, create_context
    
    router = APIRouter(prefix="/api/v1/patterns", tags=["patterns"])
    mcp = PatternsMCP()
    
    @router.post("/", response_model=PatternResponse)
    async def create_pattern(
        request_body: PatternCreateRequest,
        http_request: Request,
        caller = Depends(authenticate_caller)
    ):
        context = create_context(http_request, caller, mcp.manifest, "create")
        response = await mcp.create_pattern(request_body, context)
        return response.data  # Unwrap MCPResponse
    
    @router.post("/search", response_model=SemanticSearchResponse)
    async def search_patterns(
        request_body: SemanticSearchRequest,
        http_request: Request,
        caller = Depends(authenticate_caller)
    ):
        context = create_context(http_request, caller, mcp.manifest, "semantic_search")
        response = await mcp.semantic_search(request_body, context)
        return response.data
    
    @router.get("/{pattern_id}/metadata", response_model=PatternMetadataResponse)
    async def get_pattern_metadata(
        pattern_id: str,
        http_request: Request,
        caller = Depends(authenticate_caller)
    ):
        request_body = PatternMetadataRequest(pattern_id=pattern_id)
        context = create_context(http_request, caller, mcp.manifest, "get_metadata")
        response = await mcp.get_metadata(request_body, context)
        return response.data
    
    app.include_router(router)
    
    return mcp

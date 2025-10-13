"""Bridge to Memory/Intelligence kernels for search and queries."""
import asyncio
from datetime import timedelta
from grace.utils.time import now_utc, iso_now_utc
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MemoryBridge:
    """Bridges Interface to Memory/Intelligence kernels for librarian queries."""
    
    def __init__(self, mtl_kernel=None, intelligence_kernel=None):
        self.mtl_kernel = mtl_kernel
        self.intelligence_kernel = intelligence_kernel
        self.search_history: List[Dict] = []
    
    async def search_memory(self, query: str, user_id: str, filters: Optional[Dict] = None) -> Dict:
        """Search memory using librarian with governance-aware filtering."""
        search_id = f"search_{int(now_utc().timestamp())}"
        
        search_request = {
            "search_id": search_id,
            "query": query,
            "user_id": user_id,
            "filters": filters or {},
            "timestamp": now_utc(),
            "results": []
        }
        
        try:
            if self.mtl_kernel and hasattr(self.mtl_kernel, 'librarian'):
                # Use MTL librarian for search
                results = self.mtl_kernel.librarian.search_and_rank(
                    query=query,
                    limit=filters.get("limit", 20) if filters else 20
                )
                
                # Apply governance filtering based on user access
                filtered_results = await self._apply_governance_filters(results, user_id, filters)
                
                search_request["results"] = filtered_results
                search_request["result_count"] = len(filtered_results)
                
            else:
                logger.warning("MTL kernel or librarian not available")
                search_request["results"] = []
                search_request["error"] = "Search service unavailable"
                
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            search_request["error"] = str(e)
        
        # Store search in history
        self.search_history.append(search_request)
        
        # Keep only recent searches (last 1000)
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-1000:]
        
        return search_request
    
    async def _apply_governance_filters(self, results: List, user_id: str, filters: Optional[Dict]) -> List:
        """Apply governance-based filtering to search results."""
        if not results:
            return []
        
        filtered_results = []
        
        for result in results:
            # Check if user can access this result based on labels/trust
            if await self._can_access_result(result, user_id):
                # Apply redactions if needed
                redacted_result = await self._apply_redactions(result, user_id)
                filtered_results.append(redacted_result)
        
        return filtered_results
    
    async def _can_access_result(self, result: Any, user_id: str) -> bool:
        """Check if user can access a search result."""
        # Simple access check - in production would use proper governance
        
        # Get result metadata
        if hasattr(result, 'metadata'):
            metadata = result.metadata
        elif isinstance(result, dict):
            metadata = result.get('metadata', {})
        else:
            return True  # Allow access if no metadata
        
        # Check access level requirements
        required_access = metadata.get('access_level', 0)
        
        # Simplified: allow access for now
        # In production: check user clearance level
        return required_access <= 2
    
    async def _apply_redactions(self, result: Any, user_id: str) -> Any:
        """Apply content redactions based on user permissions."""
        # Simple redaction - in production would be more sophisticated
        
        if isinstance(result, dict):
            redacted = result.copy()
            
            # Redact sensitive fields if user doesn't have access
            sensitive_fields = ['internal_notes', 'debug_info', 'system_data']
            
            for field in sensitive_fields:
                if field in redacted:
                    redacted[field] = "[REDACTED]"
            
            return redacted
        
        return result
    
    async def get_memory_facets(self, query: str, user_id: str) -> Dict:
        """Get faceted search options for memory query."""
        facets = {
            "content_types": ["text", "code", "document", "log"],
            "time_ranges": ["last_day", "last_week", "last_month", "last_year"],
            "trust_levels": ["high", "medium", "low"],
            "sources": []
        }
        
        try:
            if self.mtl_kernel and hasattr(self.mtl_kernel, 'librarian'):
                # Get available sources/categories
                # This would be implemented in the librarian
                pass
                
        except Exception as e:
            logger.error(f"Failed to get facets: {e}")
        
        return facets
    
    async def distill_content(self, results: List[Any], context: str, user_id: str) -> Optional[str]:
        """Distill search results into a summary."""
        if not results or not self.mtl_kernel:
            return None
        
        try:
            if hasattr(self.mtl_kernel, 'librarian'):
                summary = self.mtl_kernel.librarian.distill_content(results, context)
                return summary
                
        except Exception as e:
            logger.error(f"Content distillation failed: {e}")
        
        return None
    
    def get_search_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent search history for user."""
        user_searches = [
            s for s in self.search_history
            if s.get("user_id") == user_id
        ]
        
        # Sort by timestamp descending
        user_searches.sort(key=lambda s: s["timestamp"], reverse=True)
        
        return user_searches[:limit]
    
    async def request_intel(self, request: Dict, user_id: str) -> Dict:
        """Request intelligence analysis."""
        intel_id = f"intel_{int(now_utc().timestamp())}"
        
        intel_request = {
            "intel_id": intel_id,
            "request": request,
            "user_id": user_id,
            "timestamp": now_utc(),
            "status": "pending"
        }
        
        try:
            if self.intelligence_kernel:
                # Submit to intelligence kernel
                result = await self._submit_intel_request(request)
                intel_request["result"] = result
                intel_request["status"] = "completed"
            else:
                intel_request["status"] = "error"
                intel_request["error"] = "Intelligence kernel unavailable"
                
        except Exception as e:
            logger.error(f"Intel request failed: {e}")
            intel_request["status"] = "error"
            intel_request["error"] = str(e)
        
        return intel_request
    
    async def _submit_intel_request(self, request: Dict) -> Dict:
        """Submit request to intelligence kernel."""
        # Placeholder - would use actual intelligence kernel API
        return {
            "analysis": "Intelligence analysis placeholder",
            "confidence": 0.8,
            "sources": []
        }
    
    def set_mtl_kernel(self, mtl_kernel):
        """Set MTL kernel reference."""
        self.mtl_kernel = mtl_kernel
        logger.info("MTL kernel connected to memory bridge")
    
    def set_intelligence_kernel(self, intelligence_kernel):
        """Set intelligence kernel reference."""
        self.intelligence_kernel = intelligence_kernel
        logger.info("Intelligence kernel connected to memory bridge")
    
    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        return {
            "total_searches": len(self.search_history),
            "mtl_connected": bool(self.mtl_kernel),
            "intelligence_connected": bool(self.intelligence_kernel)
        }
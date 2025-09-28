"""
Enhanced Memory Bridge with Vector Database and Quantum-Safe Storage Integration.

Provides:
- Unified interface to traditional, vector, and quantum-safe storage
- Intelligent routing based on data type and security requirements
- Enhanced recall with semantic search capabilities
- Quantum-safe encryption for sensitive governance data
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import hashlib

from .vector_db import VectorMemoryCore, MilvusVectorDB, PineconeVectorDB, EmbeddingService
from .quantum_safe_storage import QuantumSafeStorageLayer, QuantumSafeKeyManager
from ..core.memory_core import MemoryCore

logger = logging.getLogger(__name__)


class EnhancedMemoryBridge:
    """
    Enhanced memory bridge integrating multiple storage backends:
    - Traditional SQLite/PostgreSQL storage
    - Vector databases for semantic search
    - Quantum-safe encrypted storage for sensitive data
    """
    
    def __init__(self,
                 traditional_memory: Optional[MemoryCore] = None,
                 vector_memory: Optional[VectorMemoryCore] = None,
                 quantum_storage: Optional[QuantumSafeStorageLayer] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.config = config or self._get_default_config()
        
        # Initialize storage backends
        self.traditional_memory = traditional_memory or MemoryCore()
        self.vector_memory = vector_memory
        self.quantum_storage = quantum_storage
        
        # Storage routing rules
        self.routing_rules = self._initialize_routing_rules()
        
        # Statistics
        self.request_count = 0
        self.traditional_requests = 0
        self.vector_requests = 0
        self.quantum_requests = 0
        self.hybrid_requests = 0
        
        self.initialized = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the enhanced memory bridge."""
        return {
            "vector_db": {
                "enabled": True,
                "provider": "milvus",  # or "pinecone"
                "milvus_host": "localhost",
                "milvus_port": 19530,
                "pinecone_api_key": None,
                "pinecone_environment": "us-west1-gcp",
                "embedding_model": "all-MiniLM-L6-v2",
                "default_collection": "grace_memory"
            },
            "quantum_storage": {
                "enabled": True,
                "storage_path": "/tmp/grace_quantum_storage",
                "key_rotation_days": 90,
                "default_algorithm": "AES-256-GCM"
            },
            "routing": {
                "governance_to_quantum": True,
                "unstructured_to_vector": True,
                "structured_to_traditional": True,
                "sensitive_threshold": 0.8
            },
            "hybrid_search": {
                "enabled": True,
                "semantic_weight": 0.7,
                "traditional_weight": 0.3
            }
        }
    
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize storage routing rules based on data characteristics."""
        routing_config = self.config.get("routing", {})
        return {
            "governance_decisions": {
                "primary": "quantum" if routing_config.get("governance_to_quantum", True) else "traditional",
                "secondary": "vector",
                "encryption_required": True
            },
            "constitutional_data": {
                "primary": "quantum",
                "secondary": "traditional",
                "encryption_required": True
            },
            "unstructured_text": {
                "primary": "vector" if routing_config.get("unstructured_to_vector", True) else "traditional",
                "secondary": "traditional",
                "encryption_required": False
            },
            "structured_data": {
                "primary": "traditional",
                "secondary": "vector",
                "encryption_required": False
            },
            "sensitive_data": {
                "primary": "quantum",
                "secondary": None,
                "encryption_required": True
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize all storage backends."""
        try:
            logger.info("Initializing Enhanced Memory Bridge...")
            
            # Initialize traditional memory
            if hasattr(self.traditional_memory, 'start'):
                await self.traditional_memory.start()
            
            # Initialize vector memory if enabled
            if self.config["vector_db"]["enabled"]:
                if not self.vector_memory:
                    self.vector_memory = await self._create_vector_memory()
                
                if self.vector_memory and not await self.vector_memory.initialize():
                    logger.warning("Failed to initialize vector memory, continuing without it")
                    self.vector_memory = None
            
            # Initialize quantum storage if enabled
            if self.config["quantum_storage"]["enabled"]:
                if not self.quantum_storage:
                    self.quantum_storage = QuantumSafeStorageLayer(
                        storage_path=self.config["quantum_storage"]["storage_path"]
                    )
                
                if not await self.quantum_storage.initialize():
                    logger.warning("Failed to initialize quantum storage, continuing without it")
                    self.quantum_storage = None
            
            self.initialized = True
            logger.info("Enhanced Memory Bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Memory Bridge: {e}")
            return False
    
    async def _create_vector_memory(self) -> Optional[VectorMemoryCore]:
        """Create vector memory instance based on configuration."""
        try:
            config = self.config["vector_db"]
            
            # Create embedding service
            embedding_service = EmbeddingService(config["embedding_model"])
            
            # Create vector database
            if config["provider"] == "milvus":
                vector_db = MilvusVectorDB(
                    host=config["milvus_host"],
                    port=config["milvus_port"]
                )
            elif config["provider"] == "pinecone":
                if not config.get("pinecone_api_key"):
                    logger.error("Pinecone API key not provided")
                    return None
                
                vector_db = PineconeVectorDB(
                    api_key=config["pinecone_api_key"],
                    environment=config["pinecone_environment"]
                )
            else:
                logger.error(f"Unsupported vector DB provider: {config['provider']}")
                return None
            
            return VectorMemoryCore(
                vector_db=vector_db,
                embedding_service=embedding_service,
                default_collection=config["default_collection"]
            )
            
        except Exception as e:
            logger.error(f"Failed to create vector memory: {e}")
            return None
    
    def _determine_storage_strategy(self, content_type: str, data: Any, 
                                  sensitivity_score: float = 0.0) -> Dict[str, Any]:
        """Determine the best storage strategy for the given data."""
        
        # Get routing config with defaults
        routing_config = self.config.get("routing", {})
        sensitive_threshold = routing_config.get("sensitive_threshold", 0.8)
        
        # Check for sensitive data
        if sensitivity_score >= sensitive_threshold:
            return self.routing_rules["sensitive_data"]
        
        # Route based on content type
        if content_type in ["governance_decision", "constitutional_principle"]:
            return self.routing_rules["governance_decisions"]
        elif content_type in ["constitutional_data", "trust_score", "audit_log"]:
            return self.routing_rules["constitutional_data"]
        elif content_type in ["text", "document", "unstructured"]:
            return self.routing_rules["unstructured_text"]
        else:
            return self.routing_rules["structured_data"]
    
    async def store_enhanced(self,
                           content: Union[str, Dict[str, Any]],
                           content_id: Optional[str] = None,
                           content_type: str = "unstructured",
                           metadata: Optional[Dict[str, Any]] = None,
                           trust_score: float = 0.5,
                           constitutional_score: float = 0.5,
                           sensitivity_score: float = 0.0) -> Dict[str, Any]:
        """
        Store content using enhanced storage with intelligent routing.
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory bridge not initialized"}
        
        try:
            self.request_count += 1
            
            # Generate content ID if not provided
            if not content_id:
                content_hash = hashlib.sha256(str(content).encode()).hexdigest()[:16]
                content_id = f"enh_{content_type}_{content_hash}_{int(datetime.now().timestamp())}"
            
            # Determine storage strategy
            storage_strategy = self._determine_storage_strategy(
                content_type, content, sensitivity_score
            )
            
            # Prepare enhanced metadata
            enhanced_metadata = {
                "content_type": content_type,
                "trust_score": trust_score,
                "constitutional_score": constitutional_score,
                "sensitivity_score": sensitivity_score,
                "storage_strategy": storage_strategy,
                "stored_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            storage_results = {}
            
            # Store in primary storage
            primary_storage = storage_strategy["primary"]
            
            if primary_storage == "quantum" and self.quantum_storage:
                self.quantum_requests += 1
                result = await self.quantum_storage.store_encrypted(
                    data=content,
                    storage_id=content_id,
                    key_id="grace_memory",
                    metadata=enhanced_metadata
                )
                storage_results["quantum"] = result
                
            elif primary_storage == "vector" and self.vector_memory:
                self.vector_requests += 1
                result = await self.vector_memory.store_content(
                    content=str(content),
                    content_id=content_id,
                    metadata=enhanced_metadata,
                    trust_score=trust_score,
                    constitutional_score=constitutional_score
                )
                storage_results["vector"] = {"status": "success" if result else "error"}
                
            else:  # traditional storage
                self.traditional_requests += 1
                memory_id = await self.traditional_memory.store_structured_memory(
                    memory_type=content_type,
                    content={"data": content, "metadata": enhanced_metadata},
                    metadata=enhanced_metadata,
                    importance_score=(trust_score + constitutional_score) / 2
                )
                storage_results["traditional"] = {
                    "status": "success" if memory_id else "error",
                    "memory_id": memory_id
                }
            
            # Store in secondary storage if specified and different from primary
            secondary_storage = storage_strategy.get("secondary")
            if secondary_storage and secondary_storage != primary_storage:
                
                if secondary_storage == "vector" and self.vector_memory:
                    await self.vector_memory.store_content(
                        content=str(content),
                        content_id=f"{content_id}_secondary",
                        metadata=enhanced_metadata,
                        trust_score=trust_score,
                        constitutional_score=constitutional_score
                    )
                    storage_results["vector_secondary"] = {"status": "success"}
                    
                elif secondary_storage == "traditional":
                    memory_id = await self.traditional_memory.store_structured_memory(
                        memory_type=f"{content_type}_secondary",
                        content={"data": content, "metadata": enhanced_metadata},
                        metadata=enhanced_metadata,
                        importance_score=(trust_score + constitutional_score) / 2
                    )
                    storage_results["traditional_secondary"] = {
                        "status": "success" if memory_id else "error"
                    }
            
            logger.debug(f"Stored content with enhanced routing: {content_id}")
            
            return {
                "content_id": content_id,
                "primary_storage": primary_storage,
                "secondary_storage": secondary_storage,
                "storage_results": storage_results,
                "stored_at": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Enhanced storage failed for {content_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def recall_enhanced(self,
                            content_id: str,
                            search_strategy: str = "auto",
                            include_similar: bool = False,
                            similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Recall content using enhanced search across all storage backends.
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory bridge not initialized"}
        
        try:
            self.request_count += 1
            
            results = {"primary": None, "secondary": None, "similar": []}
            
            if search_strategy in ["auto", "quantum"] and self.quantum_storage:
                quantum_result = await self.quantum_storage.retrieve_encrypted(content_id)
                if quantum_result["status"] == "success":
                    results["primary"] = {
                        "source": "quantum",
                        "data": quantum_result["data"],
                        "metadata": quantum_result["metadata"],
                        "retrieved_at": quantum_result["retrieved_at"]
                    }
            
            if not results["primary"] and search_strategy in ["auto", "vector"] and self.vector_memory:
                # Try direct vector lookup first
                vector_results = await self.vector_memory.semantic_search(
                    query=content_id,  # Use ID as query for exact match
                    top_k=1
                )
                
                if vector_results:
                    results["primary"] = {
                        "source": "vector",
                        "data": vector_results[0]["content"],
                        "metadata": vector_results[0]["metadata"],
                        "score": vector_results[0]["score"],
                        "retrieved_at": datetime.now().isoformat()
                    }
            
            if not results["primary"] and search_strategy in ["auto", "traditional"]:
                memory_result = await self.traditional_memory.recall_structured_memory(content_id)
                if memory_result:
                    results["primary"] = {
                        "source": "traditional",
                        "data": memory_result.get("content"),
                        "metadata": memory_result,
                        "retrieved_at": datetime.now().isoformat()
                    }
            
            # Find similar content if requested and vector memory available
            if include_similar and self.vector_memory and results["primary"]:
                primary_data = results["primary"]["data"]
                if isinstance(primary_data, str):
                    similar_results = await self.vector_memory.semantic_search(
                        query=primary_data,
                        top_k=5
                    )
                    
                    # Filter by similarity threshold and exclude exact match
                    results["similar"] = [
                        r for r in similar_results 
                        if r["score"] >= similarity_threshold and r["id"] != content_id
                    ]
            
            if results["primary"]:
                logger.debug(f"Successfully recalled content: {content_id}")
                return {
                    "content_id": content_id,
                    "found": True,
                    "results": results,
                    "status": "success"
                }
            else:
                return {
                    "content_id": content_id,
                    "found": False,
                    "status": "not_found"
                }
            
        except Exception as e:
            logger.error(f"Enhanced recall failed for {content_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def search_enhanced(self,
                            query: str,
                            search_type: str = "hybrid",
                            top_k: int = 10,
                            min_trust_score: float = 0.0,
                            min_constitutional_score: float = 0.0,
                            content_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform enhanced search across all storage backends.
        """
        if not self.initialized:
            return {"status": "error", "error": "Memory bridge not initialized"}
        
        try:
            self.request_count += 1
            
            if search_type == "hybrid":
                self.hybrid_requests += 1
                return await self._hybrid_search(
                    query, top_k, min_trust_score, min_constitutional_score, content_types
                )
            elif search_type == "semantic" and self.vector_memory:
                self.vector_requests += 1
                results = await self.vector_memory.semantic_search(
                    query=query,
                    top_k=top_k,
                    min_trust_score=min_trust_score,
                    min_constitutional_score=min_constitutional_score
                )
                
                return {
                    "query": query,
                    "search_type": "semantic",
                    "results": results,
                    "total_results": len(results),
                    "status": "success"
                }
            else:
                # Traditional search (limited functionality)
                self.traditional_requests += 1
                # This would require implementing search in MemoryCore
                return {
                    "query": query,
                    "search_type": "traditional",
                    "results": [],
                    "total_results": 0,
                    "status": "limited_functionality"
                }
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _hybrid_search(self,
                           query: str,
                           top_k: int,
                           min_trust_score: float,
                           min_constitutional_score: float,
                           content_types: Optional[List[str]]) -> Dict[str, Any]:
        """Perform hybrid search combining semantic and traditional approaches."""
        try:
            results = {"semantic": [], "traditional": [], "combined": []}
            
            # Semantic search
            if self.vector_memory:
                semantic_results = await self.vector_memory.semantic_search(
                    query=query,
                    top_k=top_k * 2,  # Get more for better combination
                    min_trust_score=min_trust_score,
                    min_constitutional_score=min_constitutional_score
                )
                results["semantic"] = semantic_results
            
            # Traditional keyword search would go here
            # For now, we'll use semantic results only
            results["traditional"] = []
            
            # Combine results using hybrid approach
            if self.vector_memory:
                combined_results = await self.vector_memory.hybrid_search(
                    query=query,
                    keyword_results=results["traditional"],
                    top_k=top_k,
                    semantic_weight=self.config["hybrid_search"]["semantic_weight"]
                )
                results["combined"] = combined_results
            else:
                results["combined"] = results["traditional"]
            
            # Filter by content types if specified
            if content_types:
                filtered_results = []
                for result in results["combined"]:
                    metadata = result.get("metadata", {})
                    result_content_type = metadata.get("content_type", "unknown")
                    if result_content_type in content_types:
                        filtered_results.append(result)
                results["combined"] = filtered_results
            
            return {
                "query": query,
                "search_type": "hybrid",
                "results": results["combined"],
                "semantic_results": len(results["semantic"]),
                "traditional_results": len(results["traditional"]),
                "total_results": len(results["combined"]),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def maintenance_operations(self) -> Dict[str, Any]:
        """Perform maintenance operations across all storage backends."""
        try:
            maintenance_results = {}
            
            # Quantum storage key rotation
            if self.quantum_storage:
                rotation_result = await self.quantum_storage.rotate_storage_keys()
                maintenance_results["quantum_key_rotation"] = rotation_result
            
            # Vector database optimization (if supported)
            if self.vector_memory:
                # This would be provider-specific optimization
                maintenance_results["vector_optimization"] = {"status": "not_implemented"}
            
            # Traditional storage maintenance
            # This could include database vacuum, index rebuilding, etc.
            maintenance_results["traditional_maintenance"] = {"status": "not_implemented"}
            
            return {
                "maintenance_completed_at": datetime.now().isoformat(),
                "results": maintenance_results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Maintenance operations failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all storage backends."""
        try:
            health_status = {
                "overall_healthy": True,
                "backends": {}
            }
            
            # Check traditional memory
            try:
                traditional_stats = self.traditional_memory.get_memory_stats()
                health_status["backends"]["traditional"] = {
                    "healthy": True,
                    "stats": traditional_stats
                }
            except Exception as e:
                health_status["backends"]["traditional"] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
            
            # Check vector memory
            if self.vector_memory:
                try:
                    vector_stats = self.vector_memory.get_stats()
                    health_status["backends"]["vector"] = {
                        "healthy": vector_stats["initialized"],
                        "stats": vector_stats
                    }
                    if not vector_stats["initialized"]:
                        health_status["overall_healthy"] = False
                except Exception as e:
                    health_status["backends"]["vector"] = {
                        "healthy": False,
                        "error": str(e)
                    }
                    health_status["overall_healthy"] = False
            
            # Check quantum storage
            if self.quantum_storage:
                quantum_health = await self.quantum_storage.health_check()
                health_status["backends"]["quantum"] = quantum_health
                if not quantum_health.get("healthy", False):
                    health_status["overall_healthy"] = False
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"overall_healthy": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the enhanced memory bridge."""
        stats = {
            "initialized": self.initialized,
            "total_requests": self.request_count,
            "traditional_requests": self.traditional_requests,
            "vector_requests": self.vector_requests,
            "quantum_requests": self.quantum_requests,
            "hybrid_requests": self.hybrid_requests,
            "backends_enabled": {
                "traditional": True,
                "vector": self.vector_memory is not None,
                "quantum": self.quantum_storage is not None
            },
            "configuration": self.config
        }
        
        # Add backend-specific stats
        try:
            if self.traditional_memory:
                stats["traditional_stats"] = self.traditional_memory.get_memory_stats()
        except:
            pass
        
        try:
            if self.vector_memory:
                stats["vector_stats"] = self.vector_memory.get_stats()
        except:
            pass
        
        try:
            if self.quantum_storage:
                stats["quantum_stats"] = self.quantum_storage.get_stats()
        except:
            pass
        
        return stats
    
    async def shutdown(self) -> None:
        """Shutdown all storage backends."""
        try:
            if self.traditional_memory and hasattr(self.traditional_memory, 'stop'):
                await self.traditional_memory.stop()
            
            if self.vector_memory:
                await self.vector_memory.shutdown()
            
            # Quantum storage doesn't need explicit shutdown
            
            self.initialized = False
            logger.info("Enhanced Memory Bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
"""
Merkle Tree Log (MTL) - Immutable audit logs with vector search
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImmutableLogs:
    """
    Immutable audit log system with cryptographic verification and vector search
    """
    
    def __init__(self, storage_path: Optional[str] = None, embedding_service=None, vector_store=None):
        """
        Initialize immutable logs
        
        Args:
            storage_path: Path to store log files
            embedding_service: EmbeddingService instance for vectorization
            vector_store: VectorStore instance for semantic search
        """
        self.storage_path = Path(storage_path) if storage_path else Path("./data/immutable_logs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        # Initialize services if not provided
        if not self.embedding_service:
            try:
                from grace.embeddings.service import EmbeddingService
                self.embedding_service = EmbeddingService()
                logger.info("Initialized embedding service for logs")
            except Exception as e:
                logger.warning(f"Could not initialize embedding service: {e}")
        
        if not self.vector_store:
            try:
                from grace.vectorstore.service import VectorStoreService
                dimension = self.embedding_service.dimension if self.embedding_service else 384
                self.vector_store = VectorStoreService(
                    dimension=dimension,
                    index_path=str(self.storage_path / "log_vectors.bin")
                )
                logger.info("Initialized vector store for logs")
            except Exception as e:
                logger.warning(f"Could not initialize vector store: {e}")
        
        # In-memory merkle tree (simplified - production would use persistent storage)
        self.log_entries: List[Dict[str, Any]] = []
        self.previous_hash = "0" * 64  # Genesis hash
        
        logger.info(f"Immutable logs initialized at {self.storage_path}")
    
    def log_constitutional_operation(
        self,
        operation_type: str,
        actor: str,
        action: Dict[str, Any],
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Log a constitutional operation with automatic vectorization and indexing
        
        Args:
            operation_type: Type of operation (e.g., 'policy_approval', 'access_grant')
            actor: Who performed the action
            action: Action details
            result: Result of the action
            metadata: Additional metadata
            severity: Log severity (info, warning, error, critical)
            tags: List of tags for categorization
            
        Returns:
            Content ID (CID) of the log entry
        """
        timestamp = datetime.now(timezone.utc)
        
        # Create log entry
        entry = {
            "operation_type": operation_type,
            "actor": actor,
            "action": action,
            "result": result,
            "metadata": metadata or {},
            "severity": severity,
            "tags": tags or [],
            "timestamp": timestamp.isoformat(),
            "previous_hash": self.previous_hash
        }
        
        # Calculate content hash (CID)
        entry_json = json.dumps(entry, sort_keys=True)
        cid = hashlib.sha256(entry_json.encode()).hexdigest()
        entry["cid"] = cid
        
        # Calculate signature (hash of entry + previous hash)
        signature_input = f"{entry_json}{self.previous_hash}"
        entry["signature"] = hashlib.sha256(signature_input.encode()).hexdigest()
        
        # Store entry
        self.log_entries.append(entry)
        self.previous_hash = entry["signature"]
        
        # Persist to disk
        self._persist_entry(entry)
        
        logger.info(f"Logged operation: {operation_type} by {actor} (CID: {cid[:16]}...)")
        
        # Vectorize and index asynchronously
        try:
            self._vectorize_and_index(entry)
        except Exception as e:
            logger.error(f"Failed to vectorize log entry {cid}: {e}")
            # Don't fail the logging operation if vectorization fails
        
        return cid
    
    def _create_text_summary(self, entry: Dict[str, Any]) -> str:
        """
        Create a searchable text summary of the log entry
        
        Args:
            entry: Log entry dictionary
            
        Returns:
            Text summary for embedding
        """
        parts = [
            f"Operation: {entry['operation_type']}",
            f"Actor: {entry['actor']}",
            f"Severity: {entry['severity']}",
            f"Timestamp: {entry['timestamp']}",
        ]
        
        # Add tags
        if entry.get('tags'):
            parts.append(f"Tags: {', '.join(entry['tags'])}")
        
        # Add action summary
        action = entry.get('action', {})
        if action:
            action_desc = []
            for key, value in action.items():
                if isinstance(value, (str, int, float, bool)):
                    action_desc.append(f"{key}: {value}")
            if action_desc:
                parts.append("Action: " + "; ".join(action_desc))
        
        # Add result summary
        result = entry.get('result', {})
        if result:
            result_desc = []
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool)):
                    result_desc.append(f"{key}: {value}")
            if result_desc:
                parts.append("Result: " + "; ".join(result_desc))
        
        # Add metadata
        metadata = entry.get('metadata', {})
        if metadata:
            meta_desc = []
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta_desc.append(f"{key}: {value}")
            if meta_desc:
                parts.append("Context: " + "; ".join(meta_desc))
        
        return "\n".join(parts)
    
    def _vectorize_and_index(self, entry: Dict[str, Any]):
        """
        Vectorize log entry and store in vector database
        
        Args:
            entry: Log entry to vectorize
        """
        if not self.embedding_service or not self.vector_store:
            logger.debug("Skipping vectorization: services not available")
            return
        
        try:
            # Create searchable text
            text_summary = self._create_text_summary(entry)
            
            # Generate embedding
            embedding = self.embedding_service.embed_text(text_summary)
            
            # Prepare metadata for vector store
            vector_metadata = {
                "cid": entry["cid"],
                "operation_type": entry["operation_type"],
                "actor": entry["actor"],
                "severity": entry["severity"],
                "tags": entry.get("tags", []),
                "timestamp": entry["timestamp"],
                "signature": entry["signature"],
                "type": "immutable_log",
                "text_preview": text_summary[:200]  # Store preview for display
            }
            
            # Store in vector database using CID as vector ID
            self.vector_store.get_store().add_vectors(
                vectors=[embedding],
                metadata=[vector_metadata],
                ids=[f"log:{entry['cid']}"]
            )
            
            logger.debug(f"Vectorized and indexed log entry: {entry['cid'][:16]}...")
            
        except Exception as e:
            logger.error(f"Error in vectorize_and_index: {e}")
            raise
    
    def _persist_entry(self, entry: Dict[str, Any]):
        """Persist log entry to disk"""
        try:
            # Store in monthly files
            timestamp = datetime.fromisoformat(entry["timestamp"])
            month_key = f"{timestamp.year}-{timestamp.month:02d}"
            file_path = self.storage_path / f"logs-{month_key}.jsonl"
            
            with open(file_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        
        except Exception as e:
            logger.error(f"Failed to persist log entry: {e}")
    
    def get_log_entry(self, cid: str) -> Optional[Dict[str, Any]]:
        """
        Get a log entry by its CID
        
        Args:
            cid: Content ID
            
        Returns:
            Log entry or None if not found
        """
        for entry in self.log_entries:
            if entry["cid"] == cid:
                return entry.copy()
        
        # Try to load from disk
        try:
            for log_file in self.storage_path.glob("logs-*.jsonl"):
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        if entry.get("cid") == cid:
                            return entry
        except Exception as e:
            logger.error(f"Error loading log entry from disk: {e}")
        
        return None
    
    def verify_chain_integrity(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of the log chain
        
        Returns:
            (is_valid, error_message)
        """
        if not self.log_entries:
            return True, None
        
        previous_hash = "0" * 64
        
        for i, entry in enumerate(self.log_entries):
            # Verify previous hash matches
            if entry.get("previous_hash") != previous_hash:
                return False, f"Hash chain broken at entry {i}: {entry.get('cid')}"
            
            # Recalculate signature
            entry_copy = entry.copy()
            signature = entry_copy.pop("signature")
            cid = entry_copy.pop("cid")
            
            entry_json = json.dumps(entry_copy, sort_keys=True)
            expected_signature = hashlib.sha256(f"{entry_json}{previous_hash}".encode()).hexdigest()
            
            if signature != expected_signature:
                return False, f"Signature mismatch at entry {i}: {cid}"
            
            previous_hash = signature
        
        return True, None
    
    def semantic_search_logs(
        self,
        query: str,
        k: int = 10,
        severity_filter: Optional[str] = None,
        operation_type_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on immutable logs
        
        Args:
            query: Search query text
            k: Number of results to return
            severity_filter: Filter by severity level
            operation_type_filter: Filter by operation type
            tag_filter: Filter by tags (matches any)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching log entries with scores
        """
        if not self.embedding_service or not self.vector_store:
            logger.warning("Cannot perform semantic search: services not available")
            return []
        
        try:
            # Vectorize query
            query_embedding = self.embedding_service.embed_text(query)
            
            # Build filter
            filter_dict = {"type": "immutable_log"}
            if severity_filter:
                filter_dict["severity"] = severity_filter
            if operation_type_filter:
                filter_dict["operation_type"] = operation_type_filter
            
            # Search vector store
            results = self.vector_store.get_store().search(
                query_vector=query_embedding,
                k=k * 2,  # Get more results for filtering
                filter=filter_dict
            )
            
            # Process results
            log_entries = []
            for vector_id, score, metadata in results:
                # Apply similarity threshold
                if score < min_similarity:
                    continue
                
                # Apply tag filter
                if tag_filter:
                    entry_tags = metadata.get("tags", [])
                    if not any(tag in entry_tags for tag in tag_filter):
                        continue
                
                # Get full log entry
                cid = metadata.get("cid")
                if cid:
                    full_entry = self.get_log_entry(cid)
                    if full_entry:
                        full_entry["search_score"] = score
                        full_entry["relevance"] = self._calculate_relevance(score)
                        log_entries.append(full_entry)
                
                if len(log_entries) >= k:
                    break
            
            logger.info(f"Semantic log search found {len(log_entries)} results")
            return log_entries
            
        except Exception as e:
            logger.error(f"Error in semantic log search: {e}")
            return []
    
    def _calculate_relevance(self, score: float) -> str:
        """Calculate relevance category from similarity score"""
        if score > 0.9:
            return "very_high"
        elif score > 0.75:
            return "high"
        elif score > 0.5:
            return "medium"
        elif score > 0.25:
            return "low"
        else:
            return "very_low"
    
    def search_by_trust_score(
        self,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search logs by trust score range
        
        Args:
            min_trust: Minimum trust score
            max_trust: Maximum trust score
            k: Maximum number of results
            
        Returns:
            List of log entries within trust range
        """
        results = []
        
        for entry in self.log_entries:
            # Calculate trust score based on chain integrity and metadata
            trust_score = self._calculate_trust_score(entry)
            
            if min_trust <= trust_score <= max_trust:
                entry_with_trust = entry.copy()
                entry_with_trust["trust_score"] = trust_score
                results.append(entry_with_trust)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _calculate_trust_score(self, entry: Dict[str, Any]) -> float:
        """
        Calculate trust score for a log entry
        
        Based on:
        - Chain integrity
        - Entry age
        - Actor verification
        - Severity
        """
        score = 1.0
        
        # Verify signature
        try:
            entry_copy = entry.copy()
            signature = entry_copy.pop("signature", "")
            cid = entry_copy.pop("cid", "")
            previous_hash = entry.get("previous_hash", "")
            
            entry_json = json.dumps(entry_copy, sort_keys=True)
            expected_signature = hashlib.sha256(f"{entry_json}{previous_hash}".encode()).hexdigest()
            
            if signature != expected_signature:
                score *= 0.5  # Signature mismatch reduces trust
        except:
            score *= 0.3
        
        # Age factor (older entries are more trusted)
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            age_days = (datetime.now(timezone.utc) - timestamp).days
            age_factor = min(1.0, age_days / 30)  # Max trust at 30 days
            score *= (0.7 + 0.3 * age_factor)
        except:
            score *= 0.8
        
        # Severity factor (critical operations have higher trust requirements)
        severity = entry.get("severity", "info")
        if severity == "critical":
            score *= 1.0
        elif severity == "error":
            score *= 0.95
        elif severity == "warning":
            score *= 0.9
        
        return score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the immutable logs"""
        if not self.log_entries:
            return {
                "total_entries": 0,
                "chain_valid": True,
                "indexed_entries": 0
            }
        
        # Count by severity
        severity_counts = {}
        for entry in self.log_entries:
            severity = entry.get("severity", "info")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by operation type
        operation_counts = {}
        for entry in self.log_entries:
            op_type = entry.get("operation_type", "unknown")
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        # Verify chain
        chain_valid, error = self.verify_chain_integrity()
        
        return {
            "total_entries": len(self.log_entries),
            "chain_valid": chain_valid,
            "chain_error": error,
            "severity_breakdown": severity_counts,
            "operation_breakdown": operation_counts,
            "indexed_entries": self.vector_store.get_store().count() if self.vector_store else 0,
            "latest_entry": self.log_entries[-1]["cid"][:16] + "..." if self.log_entries else None
        }

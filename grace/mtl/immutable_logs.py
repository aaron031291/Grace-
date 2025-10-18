"""
Immutable Logs - Blockchain-inspired audit trail with SHA256 signatures
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Single immutable log entry"""
    entry_id: str
    timestamp: datetime
    operation_type: str
    actor: str
    action: str
    data: Dict[str, Any]
    previous_hash: str
    current_hash: str
    signature: str
    constitutional_check: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImmutableLogs:
    """
    Immutable audit log system with cryptographic guarantees
    """
    
    def __init__(self):
        self.chain: List[LogEntry] = []
        self.index: Dict[str, LogEntry] = {}
        self.genesis_hash = self._create_genesis()
        logger.info("ImmutableLogs initialized")
    
    def _create_genesis(self) -> str:
        """Create genesis block hash"""
        genesis_data = {
            'genesis': True,
            'timestamp': datetime.now().isoformat(),
            'system': 'Grace MTL'
        }
        return hashlib.sha256(json.dumps(genesis_data, sort_keys=True).encode()).hexdigest()
    
    def log_constitutional_operation(
        self,
        actor: str,
        action: str,
        data: Dict[str, Any],
        constitutional_check: bool,
        metadata: Optional[Dict] = None
    ) -> LogEntry:
        """
        Log operation with constitutional validation
        """
        # Get previous hash
        previous_hash = self.chain[-1].current_hash if self.chain else self.genesis_hash
        
        # Create entry
        entry_id = f"log_{len(self.chain)}_{int(datetime.now().timestamp())}"
        timestamp = datetime.now()
        
        # Calculate current hash
        hash_data = {
            'entry_id': entry_id,
            'timestamp': timestamp.isoformat(),
            'actor': actor,
            'action': action,
            'data': data,
            'previous_hash': previous_hash,
            'constitutional_check': constitutional_check
        }
        current_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Create signature
        signature = self._create_signature(current_hash, actor)
        
        entry = LogEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            operation_type="constitutional_operation",
            actor=actor,
            action=action,
            data=data,
            previous_hash=previous_hash,
            current_hash=current_hash,
            signature=signature,
            constitutional_check=constitutional_check,
            metadata=metadata or {}
        )
        
        # Add to chain
        self.chain.append(entry)
        self.index[entry_id] = entry
        
        logger.info(f"Logged constitutional operation: {entry_id}")
        
        return entry
    
    def _create_signature(self, hash_value: str, actor: str) -> str:
        """Create cryptographic signature"""
        sig_data = f"{hash_value}:{actor}:{datetime.now().isoformat()}"
        return hashlib.sha256(sig_data.encode()).hexdigest()
    
    def ensure_audit_immutability(self) -> bool:
        """
        Verify chain immutability using SHA256
        """
        if not self.chain:
            return True
        
        # Verify genesis
        if self.chain[0].previous_hash != self.genesis_hash:
            logger.error("Genesis hash mismatch")
            return False
        
        # Verify chain integrity
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if current.previous_hash != previous.current_hash:
                logger.error(f"Chain break at entry {i}")
                return False
            
            # Verify hash
            hash_data = {
                'entry_id': current.entry_id,
                'timestamp': current.timestamp.isoformat(),
                'actor': current.actor,
                'action': current.action,
                'data': current.data,
                'previous_hash': current.previous_hash,
                'constitutional_check': current.constitutional_check
            }
            expected_hash = hashlib.sha256(
                json.dumps(hash_data, sort_keys=True).encode()
            ).hexdigest()
            
            if current.current_hash != expected_hash:
                logger.error(f"Hash mismatch at entry {i}")
                return False
        
        logger.info("Audit immutability verified")
        return True
    
    def coordinate_transparent_audit_access(
        self,
        requestor: str,
        filters: Optional[Dict] = None,
        privacy_mask: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Provide transparent audit access with privacy masks
        """
        results = []
        
        for entry in self.chain:
            # Apply filters
            if filters:
                if 'actor' in filters and entry.actor != filters['actor']:
                    continue
                if 'action' in filters and entry.action != filters['action']:
                    continue
                if 'start_time' in filters and entry.timestamp < filters['start_time']:
                    continue
                if 'end_time' in filters and entry.timestamp > filters['end_time']:
                    continue
            
            # Apply privacy mask
            entry_data = {
                'entry_id': entry.entry_id,
                'timestamp': entry.timestamp.isoformat(),
                'action': entry.action,
                'constitutional_check': entry.constitutional_check,
                'hash': entry.current_hash[:16] + "..." if privacy_mask else entry.current_hash
            }
            
            if not privacy_mask:
                entry_data.update({
                    'actor': entry.actor,
                    'data': entry.data,
                    'signature': entry.signature
                })
            else:
                entry_data['actor'] = self._mask_actor(entry.actor)
                entry_data['data_summary'] = self._mask_data(entry.data)
            
            results.append(entry_data)
        
        logger.info(f"Audit access granted to {requestor}: {len(results)} entries")
        
        return results
    
    def _mask_actor(self, actor: str) -> str:
        """Mask actor identity"""
        if len(actor) <= 4:
            return actor[0] + "***"
        return actor[:2] + "***" + actor[-2:]
    
    def _mask_data(self, data: Dict[str, Any]) -> str:
        """Create masked data summary"""
        return f"<{len(data)} fields>"
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        if not self.chain:
            return {'total_entries': 0}
        
        return {
            'total_entries': len(self.chain),
            'immutable': self.ensure_audit_immutability(),
            'first_entry': self.chain[0].timestamp.isoformat(),
            'last_entry': self.chain[-1].timestamp.isoformat(),
            'constitutional_compliant': sum(1 for e in self.chain if e.constitutional_check),
            'unique_actors': len(set(e.actor for e in self.chain))
        }
    
    async def log_entry_with_vector_indexing(
        self,
        entry_data: Dict[str, Any],
        embedding_service=None,
        vector_store=None
    ) -> str:
        """
        Log entry and index in vector database for semantic search
        
        Args:
            entry_data: Log entry data
            embedding_service: Embedding service instance
            vector_store: Vector store instance
            
        Returns:
            Entry ID
        """
        # First, create the immutable log entry
        entry_id = self.log_constitutional_operation(
            operation_type=entry_data.get("operation_type", "general"),
            actor=entry_data.get("actor", "system"),
            action=entry_data.get("action", {}),
            result=entry_data.get("result", {}),
            metadata=entry_data.get("metadata", {})
        )
        
        # If services available, create embeddings and store in vector DB
        if embedding_service and vector_store:
            try:
                # Get the full entry
                full_entry = self.get_log_entry(entry_id)
                
                # Create searchable text representation
                searchable_text = self._create_searchable_text(full_entry)
                
                # Generate embedding
                embedding = embedding_service.embed_text(searchable_text)
                
                # Prepare metadata for vector store
                vector_metadata = {
                    "log_entry_id": entry_id,
                    "operation_type": full_entry.get("operation_type"),
                    "actor": full_entry.get("actor"),
                    "timestamp": full_entry.get("timestamp"),
                    "signature": full_entry.get("signature"),
                    "type": "immutable_log"
                }
                
                # Store in vector database
                vector_store.get_store().add_vectors(
                    vectors=[embedding],
                    metadata=[vector_metadata],
                    ids=[f"log:{entry_id}"]
                )
                
                logger.info(f"Indexed log entry {entry_id} in vector database")
                
            except Exception as e:
                logger.error(f"Failed to index log entry {entry_id} in vector database: {e}")
                # Don't fail the logging operation if vector indexing fails
        
        return entry_id
    
    def _create_searchable_text(self, entry: Dict[str, Any]) -> str:
        """
        Create a searchable text representation of a log entry
        
        Args:
            entry: Log entry dictionary
            
        Returns:
            Searchable text
        """
        parts = [
            f"Operation: {entry.get('operation_type')}",
            f"Actor: {entry.get('actor')}",
            f"Timestamp: {entry.get('timestamp')}",
        ]
        
        # Add action details
        action = entry.get("action", {})
        if action:
            parts.append("Action:")
            for key, value in action.items():
                parts.append(f"  {key}: {value}")
        
        # Add result details
        result = entry.get("result", {})
        if result:
            parts.append("Result:")
            for key, value in result.items():
                parts.append(f"  {key}: {value}")
        
        # Add metadata
        metadata = entry.get("metadata", {})
        if metadata:
            parts.append("Metadata:")
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    async def semantic_search_logs(
        self,
        query: str,
        k: int = 10,
        embedding_service=None,
        vector_store=None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on immutable logs
        
        Args:
            query: Search query
            k: Number of results
            embedding_service: Embedding service
            vector_store: Vector store
            
        Returns:
            List of matching log entries
        """
        if not embedding_service or not vector_store:
            logger.warning("Cannot perform semantic search: services not available")
            return []
        
        try:
            # Vectorize query
            query_embedding = embedding_service.embed_text(query)
            
            # Search vector store
            results = vector_store.get_store().search(
                query_vector=query_embedding,
                k=k,
                filter={"type": "immutable_log"}
            )
            
            # Retrieve full log entries
            log_entries = []
            for vector_id, score, metadata in results:
                log_id = metadata.get("log_entry_id")
                if log_id:
                    entry = self.get_log_entry(log_id)
                    if entry:
                        entry["search_score"] = score
                        log_entries.append(entry)
            
            logger.info(f"Semantic search found {len(log_entries)} matching log entries")
            return log_entries
            
        except Exception as e:
            logger.error(f"Error in semantic log search: {e}")
            return []

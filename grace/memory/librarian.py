"""
Enhanced Librarian - Document processing with chunking, semantic indexing, and constitutional filters.

Features:
- Intelligent text chunking
- Semantic embeddings
- Constitutional content filtering
- Trust scoring
- Searchable knowledge base
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .lightning import LightningMemory
from .fusion import FusionMemory

logger = logging.getLogger(__name__)


class TextChunk:
    """Individual text chunk with metadata."""
    
    def __init__(self, 
                 content: str, 
                 chunk_id: str = None,
                 source_id: str = None,
                 position: int = 0,
                 metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id or f"chunk_{uuid.uuid4().hex[:8]}"
        self.content = content
        self.source_id = source_id
        self.position = position
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        
        # Calculate characteristics
        self.word_count = len(content.split())
        self.char_count = len(content)
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_id": self.source_id,
            "position": self.position,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class ConstitutionalFilter:
    """Constitutional content filtering."""
    
    def __init__(self):
        # Define constitutional principles
        self.forbidden_patterns = [
            r'\bharm\b.*\bpeople\b',
            r'\billegal\b.*\bactivit(y|ies)\b',
            r'\bmanipulat(e|ion)\b.*\busers?\b',
            r'\bdeceiv(e|ing)\b.*\busers?\b'
        ]
        
        self.warning_patterns = [
            r'\bprivate\b.*\bdata\b',
            r'\bsensitive\b.*\binformation\b',
            r'\bunauthorized\b.*\baccess\b'
        ]
        
        self.quality_patterns = [
            r'\baccurate\b',
            r'\bverified\b',
            r'\breliable\b',
            r'\btrusted\b'
        ]
    
    def evaluate_content(self, content: str) -> Dict[str, Any]:
        """Evaluate content against constitutional principles."""
        content_lower = content.lower()
        
        # Check for forbidden content
        forbidden_violations = []
        for pattern in self.forbidden_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                forbidden_violations.extend(matches)
        
        # Check for warning patterns
        warnings = []
        for pattern in self.warning_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                warnings.extend(matches)
        
        # Check for quality indicators
        quality_score = 0
        for pattern in self.quality_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            quality_score += len(matches)
        
        # Calculate overall constitutional score
        constitutional_score = max(0, 1.0 - (len(forbidden_violations) * 0.5) - (len(warnings) * 0.2))
        constitutional_score = min(1.0, constitutional_score + (quality_score * 0.1))
        
        return {
            "constitutional_score": constitutional_score,
            "forbidden_violations": forbidden_violations,
            "warnings": warnings,
            "quality_score": quality_score,
            "approved": constitutional_score >= 0.7 and len(forbidden_violations) == 0
        }


class SemanticIndexer:
    """Simple semantic indexing without external dependencies."""
    
    def __init__(self):
        # Simple keyword-based indexing for now
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'would'
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords with simple TF scoring."""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate TF scores
        total_words = len(words)
        keyword_scores = []
        
        for word, freq in word_freq.items():
            tf_score = freq / total_words if total_words > 0 else 0
            keyword_scores.append((word, tf_score))
        
        # Sort by score and return top keywords
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:max_keywords]
    
    def create_index(self, chunks: List[TextChunk]) -> Dict[str, List[str]]:
        """Create inverted index from chunks."""
        index = {}
        
        for chunk in chunks:
            keywords = self.extract_keywords(chunk.content)
            
            for keyword, score in keywords:
                if keyword not in index:
                    index[keyword] = []
                
                # Store chunk ID with relevance score
                index[keyword].append((chunk.chunk_id, score))
        
        # Sort by relevance for each keyword
        for keyword in index:
            index[keyword].sort(key=lambda x: x[1], reverse=True)
        
        return index


class EnhancedLibrarian:
    """
    Enhanced document librarian with chunking, semantic indexing, and constitutional filtering.
    
    Processes documents into searchable chunks with trust scoring and constitutional validation.
    """
    
    def __init__(self, 
                 lightning_memory: Optional[LightningMemory] = None,
                 fusion_memory: Optional[FusionMemory] = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.lightning = lightning_memory or LightningMemory()
        self.fusion = fusion_memory or FusionMemory()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Components
        self.constitutional_filter = ConstitutionalFilter()
        self.semantic_indexer = SemanticIndexer()
        
        # In-memory indexes
        self.keyword_index = {}
        self.chunk_registry = {}
        
        logger.info("Enhanced Librarian initialized")
    
    def ingest_document(self, 
                       content: str, 
                       source_id: str = None,
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest and process a document through the complete pipeline.
        
        Returns:
            Ingestion result with statistics and chunk IDs
        """
        try:
            source_id = source_id or f"doc_{uuid.uuid4().hex[:8]}"
            
            # Step 1: Constitutional filtering
            constitutional_result = self.constitutional_filter.evaluate_content(content)
            
            if not constitutional_result["approved"]:
                logger.warning(f"Document {source_id} failed constitutional review")
                return {
                    "source_id": source_id,
                    "status": "rejected",
                    "reason": "Constitutional violations",
                    "violations": constitutional_result["forbidden_violations"],
                    "constitutional_score": constitutional_result["constitutional_score"]
                }
            
            # Step 2: Text chunking
            chunks = self._chunk_text(content, source_id)
            
            if not chunks:
                return {
                    "source_id": source_id,
                    "status": "failed",
                    "reason": "No valid chunks created"
                }
            
            # Step 3: Process each chunk
            processed_chunks = []
            failed_chunks = 0
            
            for chunk in chunks:
                try:
                    # Apply constitutional filter to chunk
                    chunk_constitutional = self.constitutional_filter.evaluate_content(chunk.content)
                    chunk.metadata.update({
                        "constitutional_score": chunk_constitutional["constitutional_score"],
                        "quality_score": chunk_constitutional["quality_score"]
                    })
                    
                    # Skip chunks with low constitutional score
                    if chunk_constitutional["constitutional_score"] < 0.5:
                        failed_chunks += 1
                        continue
                    
                    # Store in Lightning (cache) for fast access
                    cache_key = f"chunk:{chunk.chunk_id}"
                    self.lightning.put(
                        cache_key, 
                        chunk.to_dict(), 
                        ttl_seconds=3600,
                        tags=["chunk", source_id]
                    )
                    
                    # Store in Fusion (long-term) for durability
                    fusion_entry_id = self.fusion.write(
                        key=cache_key,
                        value=chunk.to_dict(),
                        content_type="application/json",
                        tags=["chunk", source_id, "processed"],
                        metadata={
                            "source_id": source_id,
                            "chunk_position": chunk.position,
                            "constitutional_score": chunk_constitutional["constitutional_score"]
                        }
                    )
                    
                    chunk.metadata["fusion_entry_id"] = fusion_entry_id
                    processed_chunks.append(chunk)
                    
                    # Update chunk registry
                    self.chunk_registry[chunk.chunk_id] = {
                        "chunk": chunk,
                        "cache_key": cache_key,
                        "fusion_entry_id": fusion_entry_id
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
                    failed_chunks += 1
            
            # Step 4: Update semantic index
            self._update_semantic_index(processed_chunks)
            
            # Step 5: Store document metadata
            doc_metadata = {
                "source_id": source_id,
                "ingested_at": datetime.utcnow().isoformat(),
                "total_chunks": len(processed_chunks),
                "failed_chunks": failed_chunks,
                "constitutional_score": constitutional_result["constitutional_score"],
                "quality_score": constitutional_result["quality_score"],
                "chunk_ids": [c.chunk_id for c in processed_chunks],
                "original_metadata": metadata or {}
            }
            
            # Store document metadata in Fusion
            doc_key = f"document:{source_id}"
            doc_entry_id = self.fusion.write(
                key=doc_key,
                value=doc_metadata,
                content_type="application/json",
                tags=["document", "metadata"],
                metadata={"document_type": "metadata"}
            )
            
            # Cache document metadata
            self.lightning.put(
                doc_key,
                doc_metadata,
                ttl_seconds=7200,
                tags=["document", source_id]
            )
            
            logger.info(f"Ingested document {source_id}: {len(processed_chunks)} chunks processed")
            
            return {
                "source_id": source_id,
                "status": "success",
                "document_entry_id": doc_entry_id,
                "chunks_processed": len(processed_chunks),
                "chunks_failed": failed_chunks,
                "constitutional_score": constitutional_result["constitutional_score"],
                "quality_score": constitutional_result["quality_score"]
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest document {source_id}: {e}")
            return {
                "source_id": source_id,
                "status": "error",
                "error": str(e)
            }
    
    def search_and_rank(self, 
                       query: str, 
                       limit: int = 10,
                       min_constitutional_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search and rank chunks by relevance with constitutional filtering.
        
        Returns:
            Ranked list of relevant chunks with trust scores
        """
        try:
            # Extract query keywords
            query_keywords = self.semantic_indexer.extract_keywords(query, max_keywords=10)
            query_terms = [kw[0] for kw in query_keywords]
            
            # Find matching chunks
            chunk_scores = {}
            
            for term in query_terms:
                if term in self.keyword_index:
                    for chunk_id, relevance_score in self.keyword_index[term]:
                        if chunk_id not in chunk_scores:
                            chunk_scores[chunk_id] = 0
                        chunk_scores[chunk_id] += relevance_score
            
            # Rank by score
            ranked_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Retrieve and filter chunks
            results = []
            for chunk_id, score in ranked_chunks[:limit * 2]:  # Get more to allow filtering
                # Try Lightning cache first
                cache_key = f"chunk:{chunk_id}"
                chunk_data = self.lightning.get(cache_key)
                
                if not chunk_data:
                    # Fallback to Fusion
                    fusion_entries = self.fusion.search(key_pattern=cache_key, limit=1)
                    if fusion_entries:
                        chunk_data = fusion_entries[0].value
                
                if chunk_data:
                    # Apply constitutional filter
                    constitutional_score = chunk_data.get("metadata", {}).get("constitutional_score", 0)
                    
                    if constitutional_score >= min_constitutional_score:
                        # Calculate trust score
                        trust_score = self._calculate_trust_score(chunk_data, score)
                        
                        results.append({
                            "chunk_id": chunk_id,
                            "content": chunk_data["content"],
                            "source_id": chunk_data.get("source_id"),
                            "relevance_score": score,
                            "constitutional_score": constitutional_score,
                            "trust_score": trust_score,
                            "metadata": chunk_data.get("metadata", {})
                        })
                
                if len(results) >= limit:
                    break
            
            logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def _chunk_text(self, text: str, source_id: str) -> List[TextChunk]:
        """Chunk text into manageable pieces with overlap."""
        chunks = []
        
        # Simple sentence-aware chunking
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    source_id=source_id,
                    position=position
                )
                chunks.append(chunk)
                
                # Handle overlap
                words = current_chunk.split()
                if len(words) > self.chunk_overlap:
                    overlap_text = " ".join(words[-self.chunk_overlap:])
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                position += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                source_id=source_id,
                position=position
            )
            chunks.append(chunk)
        
        return chunks
    
    def _update_semantic_index(self, chunks: List[TextChunk]):
        """Update the semantic index with new chunks."""
        new_index = self.semantic_indexer.create_index(chunks)
        
        # Merge with existing index
        for keyword, chunk_scores in new_index.items():
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            
            # Add new chunk scores
            existing_chunks = {chunk_id for chunk_id, _ in self.keyword_index[keyword]}
            for chunk_id, score in chunk_scores:
                if chunk_id not in existing_chunks:
                    self.keyword_index[keyword].append((chunk_id, score))
            
            # Re-sort by relevance
            self.keyword_index[keyword].sort(key=lambda x: x[1], reverse=True)
    
    def _calculate_trust_score(self, chunk_data: Dict[str, Any], relevance_score: float) -> float:
        """Calculate trust score for a chunk based on various factors."""
        constitutional_score = chunk_data.get("metadata", {}).get("constitutional_score", 0)
        quality_score = chunk_data.get("metadata", {}).get("quality_score", 0)
        
        # Simple trust calculation
        trust_score = (
            constitutional_score * 0.5 +
            min(relevance_score, 1.0) * 0.3 +
            min(quality_score / 10, 1.0) * 0.2
        )
        
        return min(trust_score, 1.0)
    
    def get_document_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document."""
        doc_key = f"document:{source_id}"
        
        # Try cache first
        doc_info = self.lightning.get(doc_key)
        if doc_info:
            return doc_info
        
        # Fallback to Fusion
        entries = self.fusion.search(key_pattern=doc_key, limit=1)
        if entries:
            doc_info = entries[0].value
            # Cache for future use
            self.lightning.put(doc_key, doc_info, ttl_seconds=3600)
            return doc_info
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get librarian statistics."""
        lightning_stats = self.lightning.get_stats()
        fusion_stats = self.fusion.get_stats()
        
        return {
            "chunks_in_registry": len(self.chunk_registry),
            "keywords_indexed": len(self.keyword_index),
            "lightning_cache": lightning_stats,
            "fusion_storage": fusion_stats,
            "uptime": "running"
        }
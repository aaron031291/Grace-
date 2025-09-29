"""
Main memory ingestion pipeline.

Coordinates file extraction, chunking, embedding, and vector storage.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
from uuid import uuid4, UUID
from pathlib import Path
import tempfile
import os

from .text_extraction import get_text_extractor
from .text_chunking import get_text_chunker, TextChunk
from .embeddings import get_embedding_generator
from .vector_store import get_vector_search_service
from ..core.database.connection import get_database
from ..core.database.repositories import RepositoryFactory

logger = logging.getLogger(__name__)


class MemoryIngestionPipeline:
    """Main pipeline for ingesting files into memory system."""
    
    def __init__(self, vector_url: Optional[str] = None):
        self.text_extractor = get_text_extractor()
        self.text_chunker = get_text_chunker()
        self.embedding_generator = get_embedding_generator()
        self.vector_service = get_vector_search_service(vector_url)
        
    async def ingest_file(self, file_path: str, 
                         session_id: Optional[UUID] = None,
                         user_id: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         trust_score: float = 0.7) -> Dict[str, Any]:
        """
        Ingest a file into the memory system.
        
        Args:
            file_path: Path to the file to ingest
            session_id: Optional session ID to associate content with
            user_id: User ID who owns the content
            tags: Optional tags for categorization
            trust_score: Trust score for the content (0.0 to 1.0)
            
        Returns:
            Ingestion result with statistics and IDs
        """
        logger.info(f"Starting ingestion for file: {file_path}")
        
        try:
            # Step 1: Extract text from file
            extraction_result = await self.text_extractor.extract_text(file_path)
            text_content = extraction_result['text']
            file_metadata = extraction_result['metadata']
            
            if not text_content.strip():
                logger.warning(f"No text content extracted from {file_path}")
                return {
                    'status': 'skipped',
                    'reason': 'No text content',
                    'file_metadata': file_metadata
                }
            
            # Step 2: Chunk the text
            chunks = self.text_chunker.chunk_text(text_content, file_metadata)
            logger.info(f"Created {len(chunks)} chunks from file")
            
            if not chunks:
                return {
                    'status': 'skipped',
                    'reason': 'No chunks created',
                    'file_metadata': file_metadata
                }
            
            # Step 3: Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embedding_results = await self.embedding_generator.generate_embeddings(chunk_texts)
            
            # Step 4: Store in database and vector store
            async with get_database() as db_session:
                repo_factory = RepositoryFactory(db_session)
                knowledge_repo = repo_factory.get_knowledge_entry_repo()
                fragment_repo = repo_factory.get_fragment_repo()
                
                # Create main knowledge entry
                knowledge_entry = await knowledge_repo.create(
                    user_id=user_id or 'system',
                    session_id=session_id,
                    title=file_metadata['file_name'],
                    content=text_content,
                    content_type=file_metadata['mime_type'],
                    content_hash=file_metadata['file_hash'],
                    trust_score=trust_score,
                    credibility_score=trust_score,  # Start with same as trust
                    source=f"file:{file_path}",
                    extra_data={
                        'file_metadata': file_metadata,
                        'ingestion_stats': {
                            'chunk_count': len(chunks),
                            'embedding_model': self.embedding_generator.get_model_name()
                        },
                        'tags': tags or []
                    }
                )
                
                # Store chunks and embeddings
                ingested_chunks = []
                for chunk, embedding_result in zip(chunks, embedding_results):
                    if 'error' in embedding_result:
                        logger.warning(f"Skipping chunk due to embedding error: {embedding_result['error']}")
                        continue
                    
                    # Store fragment in database
                    fragment = await fragment_repo.create(
                        message_id=None,  # Not associated with a message
                        content=chunk.content,
                        content_hash=chunk.chunk_hash,
                        start_pos=chunk.start_pos,
                        end_pos=chunk.end_pos,
                        embedding=embedding_result['embedding'],
                        extra_data={
                            'chunk_metadata': chunk.metadata,
                            'knowledge_entry_id': str(knowledge_entry.id),
                            'token_count': chunk.token_count,
                            'embedding_model': embedding_result['model_name']
                        }
                    )
                    
                    # Prepare for vector store
                    vector_metadata = {
                        'fragment_id': str(fragment.id),
                        'knowledge_entry_id': str(knowledge_entry.id),
                        'session_id': str(session_id) if session_id else None,
                        'user_id': user_id,
                        'content_type': file_metadata['mime_type'],
                        'file_name': file_metadata['file_name'],
                        'chunk_index': chunk.metadata['chunk_index'],
                        'trust_score': trust_score,
                        'tags': tags or [],
                        'text_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                    }
                    
                    # Index in vector store
                    await self.vector_service.index_content(
                        content_id=str(fragment.id),
                        vector=embedding_result['embedding'],
                        metadata=vector_metadata
                    )
                    
                    ingested_chunks.append({
                        'fragment_id': str(fragment.id),
                        'chunk_hash': chunk.chunk_hash,
                        'token_count': chunk.token_count,
                        'embedding_dimension': embedding_result['embedding_dimension']
                    })
                
                # Commit database transaction
                await db_session.commit()
                
                logger.info(f"Successfully ingested file {file_path} with {len(ingested_chunks)} chunks")
                
                return {
                    'status': 'success',
                    'knowledge_entry_id': str(knowledge_entry.id),
                    'chunks_ingested': len(ingested_chunks),
                    'total_chunks': len(chunks),
                    'file_metadata': file_metadata,
                    'ingested_chunks': ingested_chunks
                }
                
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'file_path': file_path
            }
    
    async def ingest_text_content(self, text: str, 
                                 title: str = "Text Content",
                                 session_id: Optional[UUID] = None,
                                 user_id: Optional[str] = None,
                                 tags: Optional[List[str]] = None,
                                 trust_score: float = 0.7) -> Dict[str, Any]:
        """
        Ingest raw text content directly.
        
        Args:
            text: Text content to ingest
            title: Title for the content
            session_id: Optional session ID
            user_id: User ID who owns the content
            tags: Optional tags
            trust_score: Trust score (0.0 to 1.0)
            
        Returns:
            Ingestion result
        """
        logger.info(f"Starting ingestion for text content: {title}")
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(text)
            tmp_path = tmp_file.name
        
        try:
            # Process as file
            result = await self.ingest_file(
                file_path=tmp_path,
                session_id=session_id,
                user_id=user_id,
                tags=tags,
                trust_score=trust_score
            )
            
            # Update metadata to reflect it's text content
            if result['status'] == 'success':
                result['content_type'] = 'direct_text'
                result['title'] = title
            
            return result
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    
    async def search_memory(self, query: str,
                          session_id: Optional[UUID] = None,
                          user_id: Optional[str] = None,
                          tags: Optional[List[str]] = None,
                          trust_threshold: float = 0.5,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memory using semantic similarity.
        
        Args:
            query: Search query text
            session_id: Optional session filter
            user_id: Optional user filter
            tags: Optional tag filters
            trust_threshold: Minimum trust score
            limit: Maximum results
            
        Returns:
            List of search results
        """
        logger.info(f"Searching memory for: {query}")
        
        try:
            # Generate embedding for query
            query_embedding_result = await self.embedding_generator.generate_embedding(query)
            query_vector = query_embedding_result['embedding']
            
            # Build filters
            filters = {}
            if session_id:
                filters['session_id'] = str(session_id)
            if user_id:
                filters['user_id'] = user_id
            # Note: tag filtering would need more sophisticated implementation in vector store
            
            # Perform vector search
            search_results = await self.vector_service.search_content(
                query_vector=query_vector,
                filters=filters,
                trust_threshold=trust_threshold,
                limit=limit
            )
            
            # Enhance results with database information
            async with get_database() as db_session:
                repo_factory = RepositoryFactory(db_session)
                fragment_repo = repo_factory.get_fragment_repo()
                knowledge_repo = repo_factory.get_knowledge_entry_repo()
                
                enhanced_results = []
                for result in search_results:
                    metadata = result['metadata']
                    fragment_id = metadata.get('fragment_id')
                    
                    if fragment_id:
                        # Get full fragment information
                        fragment = await fragment_repo.get_by_id(UUID(fragment_id))
                        if fragment:
                            enhanced_result = {
                                'fragment_id': fragment_id,
                                'content': fragment.content,
                                'similarity_score': result['score'],
                                'trust_score': metadata.get('trust_score', 0.5),
                                'metadata': metadata,
                                'content_type': metadata.get('content_type'),
                                'file_name': metadata.get('file_name'),
                                'chunk_index': metadata.get('chunk_index'),
                                'text_preview': metadata.get('text_preview')
                            }
                            enhanced_results.append(enhanced_result)
                
                logger.info(f"Found {len(enhanced_results)} relevant memory fragments")
                return enhanced_results
        
        except Exception as e:
            logger.error(f"Memory search failed: {e}", exc_info=True)
            return []


# Global pipeline instance
_pipeline = None

def get_memory_ingestion_pipeline(vector_url: Optional[str] = None) -> MemoryIngestionPipeline:
    """Get global memory ingestion pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MemoryIngestionPipeline(vector_url=vector_url)
    return _pipeline
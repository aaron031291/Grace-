"""
Book Ingestion Service for Grace - Large Document Processing
===========================================================

Specialized service for ingesting and processing large documents (books, reports, etc.)
with enhanced capabilities for 500+ page documents:

- Progressive chunking with context preservation
- Chapter/section awareness
- Long-form content summarization
- Key insights extraction
- Memory-efficient processing
- Progress tracking for large ingestion jobs

Usage:
    from grace.memory.book_ingestion import BookIngestionService
    
    service = BookIngestionService()
    result = await service.ingest_book(content, "book_title")
"""

import asyncio
import logging
import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass, asdict

from .librarian import EnhancedLibrarian, TextChunk


logger = logging.getLogger(__name__)


@dataclass
class BookChapter:
    """Represents a chapter or major section in a book."""
    chapter_id: str
    title: str
    content: str
    position: int
    word_count: int
    summary: Optional[str] = None
    key_insights: List[str] = None
    

@dataclass 
class IngestionProgress:
    """Tracks progress of book ingestion."""
    total_pages: int
    pages_processed: int
    total_chapters: int
    chapters_processed: int
    chunks_created: int
    current_phase: str
    estimated_time_remaining: Optional[float] = None
    

@dataclass
class BookInsight:
    """Represents an insight extracted from the book."""
    insight_id: str
    content: str
    context: str
    relevance_score: float
    chapter_source: str
    insight_type: str  # 'key_concept', 'actionable', 'factual', 'philosophical'


class BookIngestionService:
    """Service for processing large books and documents."""
    
    def __init__(self, librarian: Optional[EnhancedLibrarian] = None):
        self.librarian = librarian or EnhancedLibrarian()
        self.ingestion_jobs = {}  # Track ongoing jobs
        
        # Configuration for large documents
        self.config = {
            "max_chunk_size": 2000,      # Larger chunks for books
            "chunk_overlap": 200,        # Maintain context across chunks
            "min_chapter_words": 100,    # Minimum words to consider a chapter (reduced)
            "insight_threshold": 0.7,    # Minimum relevance for insights
            "progress_update_interval": 10,  # Update progress every N chunks
            "max_concurrent_chunks": 5,  # Limit parallel processing
        }
    
    async def ingest_book(self, 
                         content: str,
                         title: str,
                         author: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest a complete book with progress tracking.
        
        Args:
            content: Full text content of the book
            title: Book title
            author: Book author (optional)
            metadata: Additional metadata
            
        Returns:
            Ingestion result with job ID for tracking
        """
        job_id = f"book_{uuid.uuid4().hex[:8]}"
        
        # Estimate pages (assuming ~250 words per page)
        estimated_pages = max(1, len(content.split()) // 250)
        
        progress = IngestionProgress(
            total_pages=estimated_pages,
            pages_processed=0,
            total_chapters=0,
            chapters_processed=0,
            chunks_created=0,
            current_phase="starting"
        )
        
        self.ingestion_jobs[job_id] = progress
        
        try:
            logger.info(f"Starting book ingestion: {title} (estimated {estimated_pages} pages)")
            
            # Phase 1: Chapter detection and extraction
            progress.current_phase = "detecting_chapters"
            chapters = await self._detect_chapters(content, title)
            progress.total_chapters = len(chapters)
            
            # Phase 2: Process chapters progressively
            progress.current_phase = "processing_chapters"
            processed_chunks = []
            insights = []
            
            for i, chapter in enumerate(chapters):
                # Process chapter
                chapter_chunks = await self._process_chapter(chapter, title)
                processed_chunks.extend(chapter_chunks)
                
                # Extract insights from chapter
                chapter_insights = await self._extract_chapter_insights(chapter)
                insights.extend(chapter_insights)
                
                # Update progress
                progress.chapters_processed = i + 1
                progress.chunks_created = len(processed_chunks)
                progress.pages_processed = min(progress.total_pages, 
                                             int((i + 1) / len(chapters) * progress.total_pages))
                
                if i % 5 == 0:  # Log progress every 5 chapters
                    logger.info(f"Book ingestion progress: {progress.chapters_processed}/{progress.total_chapters} chapters")
                
                # Yield control periodically
                await asyncio.sleep(0.01)
            
            # Phase 3: Store in memory systems
            progress.current_phase = "storing_memory"
            
            # Create book metadata
            book_metadata = {
                "title": title,
                "author": author,
                "ingestion_date": datetime.utcnow().isoformat(),
                "total_chapters": len(chapters),
                "total_chunks": len(processed_chunks),
                "total_insights": len(insights),
                "word_count": len(content.split()),
                "estimated_pages": estimated_pages,
                "job_id": job_id,
                **(metadata or {})
            }
            
            # Use existing librarian to store chunks
            librarian_result = self.librarian.ingest_document(
                content=content,
                source_id=f"book_{title.replace(' ', '_').lower()}",
                metadata=book_metadata
            )
            
            # Store book-specific data
            await self._store_book_insights(insights, job_id, title)
            await self._store_chapter_summaries(chapters, job_id, title)
            
            progress.current_phase = "completed"
            
            result = {
                "job_id": job_id,
                "status": "success",
                "title": title,
                "author": author,
                "processing_summary": {
                    "chapters_processed": len(chapters),
                    "chunks_created": len(processed_chunks),
                    "insights_extracted": len(insights),
                    "total_pages": estimated_pages,
                    "word_count": len(content.split())
                },
                "librarian_result": librarian_result,
                "access_methods": {
                    "chapter_search": f"/api/books/{job_id}/chapters",
                    "insight_search": f"/api/books/{job_id}/insights",
                    "full_text_search": f"/api/mtl/memory?query={title}"
                }
            }
            
            logger.info(f"Book ingestion completed: {title} ({len(chapters)} chapters, {len(insights)} insights)")
            return result
            
        except Exception as e:
            progress.current_phase = "error"
            logger.error(f"Book ingestion failed for {title}: {e}")
            
            return {
                "job_id": job_id,
                "status": "error",
                "title": title,
                "error": str(e),
                "progress": asdict(progress)
            }
    
    async def _detect_chapters(self, content: str, title: str) -> List[BookChapter]:
        """Detect chapters or major sections in the book."""
        chapters = []
        
        # Common chapter patterns
        chapter_patterns = [
            r'^Chapter\s+\d+.*?$',           # Chapter 1, Chapter One, etc.
            r'^\d+\.\s+.*?$',               # 1. Introduction, 2. Background
            r'^[IVX]+\.\s+.*?$',           # I. Introduction, II. Methods  
            r'^Part\s+\d+.*?$',            # Part 1, Part Two
            r'^Section\s+\d+.*?$',         # Section 1, Section A
        ]
        
        # Split content into lines
        lines = content.split('\n')
        chapter_starts = []
        
        # Find chapter boundaries
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Debug: print what we're checking
            # Check if line matches chapter pattern
            for pattern in chapter_patterns:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    chapter_starts.append((i, line_clean))
                    logger.info(f"Found chapter: {line_clean}")
                    break
        
        logger.info(f"Total chapter starts found: {len(chapter_starts)}")
        
        # If no chapters detected, try more permissive patterns
        if not chapter_starts:
            logger.info("No chapters found with strict patterns, trying permissive patterns")
            
            # More permissive patterns
            permissive_patterns = [
                r'^Chapter\s*\d+',              # Chapter1, Chapter 1 
                r'^\d+\s*[.:]',                # 1. or 1:
                r'^[IVX]+[.:]',                # I. or I:
                r'Chapter\s+[IVXivx\d]+',      # Chapter I, Chapter 1
            ]
            
            for i, line in enumerate(lines):
                line_clean = line.strip()
                if not line_clean or len(line_clean) > 100:  # Skip very long lines
                    continue
                    
                for pattern in permissive_patterns:
                    if re.search(pattern, line_clean, re.IGNORECASE):
                        chapter_starts.append((i, line_clean))
                        logger.info(f"Found chapter (permissive): {line_clean}")
                        break
        
        # If still no chapters detected, treat as single document
        if not chapter_starts:
            logger.info(f"No chapters detected in {title}, treating as single document")
            return [BookChapter(
                chapter_id=f"chapter_001",
                title=title,
                content=content,
                position=0,
                word_count=len(content.split())
            )]
        
        # Remove duplicates (same line might match multiple patterns)
        chapter_starts = list(set(chapter_starts))
        chapter_starts.sort(key=lambda x: x[0])  # Sort by line number
        
        # Extract chapter content
        for i, (start_line, chapter_title) in enumerate(chapter_starts):
            # Determine end of chapter
            if i + 1 < len(chapter_starts):
                end_line = chapter_starts[i + 1][0]
            else:
                end_line = len(lines)
            
            # Extract chapter text
            chapter_lines = lines[start_line:end_line]
            chapter_content = '\n'.join(chapter_lines).strip()
            
            if len(chapter_content.split()) < self.config["min_chapter_words"]:
                logger.info(f"Skipping short chapter: {chapter_title} ({len(chapter_content.split())} words)")
                continue  # Skip very short chapters
            
            chapter = BookChapter(
                chapter_id=f"chapter_{i+1:03d}",
                title=chapter_title,
                content=chapter_content,
                position=i,
                word_count=len(chapter_content.split())
            )
            
            # Generate summary
            chapter.summary = await self._summarize_chapter(chapter_content)
            
            chapters.append(chapter)
        
        logger.info(f"Detected {len(chapters)} valid chapters in {title}")
        return chapters
    
    async def _process_chapter(self, chapter: BookChapter, book_title: str) -> List[TextChunk]:
        """Process a chapter into chunks suitable for the librarian."""
        chunks = []
        
        # Use larger chunks for books with overlap for context
        chunk_size = self.config["max_chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        words = chapter.content.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            if not chunk_content.strip():
                continue
            
            chunk = TextChunk(
                content=chunk_content,
                source_id=f"{book_title}_{chapter.chapter_id}",
                position=i // (chunk_size - overlap),
                metadata={
                    "book_title": book_title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "chapter_position": chapter.position,
                    "chunk_type": "book_content"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _summarize_chapter(self, content: str) -> str:
        """Generate a summary of the chapter content."""
        # Simple extractive summary (first few sentences + key sentences)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 3:
            return ' '.join(sentences)
        
        # Take first 2 sentences and try to find a concluding sentence
        summary_parts = sentences[:2]
        
        # Look for sentences with conclusion indicators
        for sentence in sentences[2:]:
            if any(indicator in sentence.lower() for indicator in 
                   ['therefore', 'thus', 'conclusion', 'summary', 'result']):
                summary_parts.append(sentence)
                break
        
        return '. '.join(summary_parts) + '.'
    
    async def _extract_chapter_insights(self, chapter: BookChapter) -> List[BookInsight]:
        """Extract key insights from a chapter."""
        insights = []
        content = chapter.content.lower()
        
        # Look for sentences with insight indicators
        insight_patterns = [
            (r'important.*?[.!?]', 'key_concept'),
            (r'key.*?[.!?]', 'key_concept'),
            (r'should.*?[.!?]', 'actionable'),
            (r'must.*?[.!?]', 'actionable'),
            (r'research shows.*?[.!?]', 'factual'),
            (r'studies indicate.*?[.!?]', 'factual'),
            (r'suggests that.*?[.!?]', 'factual'),
        ]
        
        sentences = re.split(r'[.!?]+', chapter.content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 50:  # Skip very short sentences
                continue
                
            for pattern, insight_type in insight_patterns:
                if re.search(pattern, sentence.lower()):
                    insight = BookInsight(
                        insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                        content=sentence,
                        context=chapter.title,
                        relevance_score=0.8,  # Could be enhanced with ML scoring
                        chapter_source=chapter.chapter_id,
                        insight_type=insight_type
                    )
                    insights.append(insight)
                    break  # Only one type per sentence
        
        return insights[:10]  # Limit insights per chapter
    
    async def _store_book_insights(self, insights: List[BookInsight], job_id: str, title: str):
        """Store book insights in memory system."""
        try:
            insights_data = {
                "job_id": job_id,
                "book_title": title,
                "insights": [asdict(insight) for insight in insights],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store in Lightning for quick access
            self.librarian.lightning.put(
                f"book_insights:{job_id}",
                insights_data,
                ttl_seconds=7200,
                tags=["book", "insights", job_id]
            )
            
            logger.info(f"Stored {len(insights)} insights for book {title}")
            
        except Exception as e:
            logger.error(f"Failed to store insights: {e}")
    
    async def _store_chapter_summaries(self, chapters: List[BookChapter], job_id: str, title: str):
        """Store chapter summaries in memory system."""
        try:
            chapters_data = {
                "job_id": job_id,
                "book_title": title,
                "chapters": [asdict(chapter) for chapter in chapters],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store in Lightning for quick access
            self.librarian.lightning.put(
                f"book_chapters:{job_id}",
                chapters_data,
                ttl_seconds=7200,
                tags=["book", "chapters", job_id]
            )
            
            logger.info(f"Stored {len(chapters)} chapter summaries for book {title}")
            
        except Exception as e:
            logger.error(f"Failed to store chapter summaries: {e}")
    
    def get_ingestion_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for an ingestion job."""
        if job_id in self.ingestion_jobs:
            return asdict(self.ingestion_jobs[job_id])
        return None
    
    async def get_book_insights(self, job_id: str) -> List[Dict[str, Any]]:
        """Get insights for a processed book."""
        try:
            data = self.librarian.lightning.get(f"book_insights:{job_id}")
            if data:
                return data.get("insights", [])
        except Exception as e:
            logger.error(f"Failed to get book insights: {e}")
        return []
    
    async def get_book_chapters(self, job_id: str) -> List[Dict[str, Any]]:
        """Get chapter summaries for a processed book."""
        try:
            data = self.librarian.lightning.get(f"book_chapters:{job_id}")
            if data:
                return data.get("chapters", [])
        except Exception as e:
            logger.error(f"Failed to get book chapters: {e}")
        return []
    
    async def search_book_content(self, job_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific book's content."""
        try:
            # Get all chunks for this book
            # This would integrate with the vector search system
            results = []
            
            # For now, search through stored insights and chapters
            insights = await self.get_book_insights(job_id)
            chapters = await self.get_book_chapters(job_id)
            
            query_lower = query.lower()
            
            # Search insights
            for insight in insights:
                if query_lower in insight.get("content", "").lower():
                    results.append({
                        "type": "insight",
                        "content": insight["content"],
                        "context": insight["context"],
                        "relevance": insight["relevance_score"]
                    })
            
            # Search chapter summaries
            for chapter in chapters:
                if query_lower in chapter.get("summary", "").lower():
                    results.append({
                        "type": "chapter_summary", 
                        "content": chapter["summary"],
                        "context": chapter["title"],
                        "relevance": 0.7
                    })
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search book content: {e}")
            return []
"""
Multi-Modal Data Ingestion System

Grace can ingest and learn from:
- Web pages (scraping, crawling)
- PDFs (documents, books, papers)
- Code repositories
- Audio files (transcription)
- Video files (transcription + vision)
- Text documents
- APIs and datasets

All ingested data flows to persistent memory and
builds Grace's autonomous knowledge.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class IngestedData:
    """Data that has been ingested"""
    data_id: str
    source_type: str
    source_url: Optional[str]
    content: str
    metadata: Dict[str, Any]
    ingested_at: datetime
    chunks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "content_length": len(self.content),
            "chunk_count": len(self.chunks),
            "ingested_at": self.ingested_at.isoformat(),
            "metadata": self.metadata
        }


class WebIngestion:
    """Ingest data from web pages"""
    
    @staticmethod
    async def scrape_url(url: str) -> IngestedData:
        """Scrape web page"""
        logger.info(f"Scraping: {url}")
        
        try:
            # In production, use requests/httpx + BeautifulSoup
            # For now, simulate
            content = f"Web content from {url}"
            
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            data_id = hashlib.sha256(url.encode()).hexdigest()[:16]
            
            return IngestedData(
                data_id=data_id,
                source_type="web",
                source_url=url,
                content=content,
                metadata={"url": url, "scraped_at": datetime.utcnow().isoformat()},
                ingested_at=datetime.utcnow(),
                chunks=chunks
            )
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            raise


class PDFIngestion:
    """Ingest data from PDF files"""
    
    @staticmethod
    async def ingest_pdf(file_path: str) -> IngestedData:
        """Extract text from PDF"""
        logger.info(f"Ingesting PDF: {file_path}")
        
        try:
            # In production, use PyPDF2 or pdfplumber
            # For now, simulate
            content = f"PDF content from {file_path}"
            
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            data_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
            
            return IngestedData(
                data_id=data_id,
                source_type="pdf",
                source_url=None,
                content=content,
                metadata={
                    "file_path": file_path,
                    "pages": 10,  # Simulated
                },
                ingested_at=datetime.utcnow(),
                chunks=chunks
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF {file_path}: {e}")
            raise


class CodeIngestion:
    """Ingest code repositories and files"""
    
    @staticmethod
    async def ingest_repository(repo_url: str) -> List[IngestedData]:
        """Ingest entire code repository"""
        logger.info(f"Ingesting repository: {repo_url}")
        
        # In production, clone repo and parse files
        # For now, simulate
        files = [
            {"path": "main.py", "content": "# Python code"},
            {"path": "app.js", "content": "// JavaScript code"}
        ]
        
        ingested = []
        for file_data in files:
            data_id = hashlib.sha256(
                f"{repo_url}{file_data['path']}".encode()
            ).hexdigest()[:16]
            
            data = IngestedData(
                data_id=data_id,
                source_type="code",
                source_url=repo_url,
                content=file_data['content'],
                metadata={
                    "repo_url": repo_url,
                    "file_path": file_data['path'],
                    "language": file_data['path'].split('.')[-1]
                },
                ingested_at=datetime.utcnow(),
                chunks=[file_data['content']]
            )
            
            ingested.append(data)
        
        return ingested


class AudioIngestion:
    """Ingest audio files (with transcription)"""
    
    @staticmethod
    async def ingest_audio(file_path: str) -> IngestedData:
        """Transcribe and ingest audio"""
        logger.info(f"Ingesting audio: {file_path}")
        
        try:
            # In production, use Whisper API or similar
            # For now, simulate
            transcript = f"Transcribed content from {file_path}"
            
            data_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
            
            return IngestedData(
                data_id=data_id,
                source_type="audio",
                source_url=None,
                content=transcript,
                metadata={
                    "file_path": file_path,
                    "duration_seconds": 120,
                    "transcription_service": "whisper"
                },
                ingested_at=datetime.utcnow(),
                chunks=[transcript]
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest audio {file_path}: {e}")
            raise


class VideoIngestion:
    """Ingest video files (transcription + vision)"""
    
    @staticmethod
    async def ingest_video(file_path: str) -> IngestedData:
        """Extract audio transcription and visual description from video"""
        logger.info(f"Ingesting video: {file_path}")
        
        try:
            # In production:
            # 1. Extract audio → Whisper transcription
            # 2. Extract frames → Vision model description
            # 3. Combine into comprehensive content
            
            content = f"Video content from {file_path}\n"
            content += "Transcript: ...\n"
            content += "Visual description: ..."
            
            data_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
            
            return IngestedData(
                data_id=data_id,
                source_type="video",
                source_url=None,
                content=content,
                metadata={
                    "file_path": file_path,
                    "duration_seconds": 300,
                    "has_transcript": True,
                    "has_visual_description": True
                },
                ingested_at=datetime.utcnow(),
                chunks=[content]
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest video {file_path}: {e}")
            raise


class MultiModalIngestionEngine:
    """
    Complete multi-modal ingestion system.
    
    Grace can learn from ANY data source.
    """
    
    def __init__(self, persistent_memory):
        self.memory = persistent_memory
        self.ingestion_history = []
        
        logger.info("Multi-Modal Ingestion Engine initialized")
    
    async def ingest(
        self,
        source_type: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestedData:
        """
        Universal ingestion method.
        
        Args:
            source_type: web, pdf, code, audio, video, text
            source: URL, file path, or text content
            metadata: Additional metadata
        
        Returns:
            IngestedData object
        """
        logger.info(f"Ingesting {source_type}: {source[:100]}...")
        
        metadata = metadata or {}
        
        # Route to appropriate ingestion handler
        if source_type == "web":
            data = await WebIngestion.scrape_url(source)
        
        elif source_type == "pdf":
            data = await PDFIngestion.ingest_pdf(source)
        
        elif source_type == "code":
            # Single file or repository
            if source.startswith("http"):
                datas = await CodeIngestion.ingest_repository(source)
                data = datas[0] if datas else None
            else:
                # Single file
                content = Path(source).read_text()
                data = await self._ingest_text(content, "code", metadata)
        
        elif source_type == "audio":
            data = await AudioIngestion.ingest_audio(source)
        
        elif source_type == "video":
            data = await VideoIngestion.ingest_video(source)
        
        elif source_type == "text":
            data = await self._ingest_text(source, "text", metadata)
        
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Store in persistent memory
        await self.memory.ingest_document(
            source_type=data.source_type,
            content=data.content,
            metadata=data.metadata
        )
        
        # Store in history
        self.ingestion_history.append({
            "data_id": data.data_id,
            "source_type": source_type,
            "ingested_at": data.ingested_at.isoformat()
        })
        
        logger.info(f"✅ Ingested: {data.data_id}")
        logger.info(f"   Type: {source_type}")
        logger.info(f"   Size: {len(data.content)} chars")
        logger.info(f"   Chunks: {len(data.chunks)}")
        
        return data
    
    async def _ingest_text(
        self,
        text: str,
        text_type: str,
        metadata: Dict[str, Any]
    ) -> IngestedData:
        """Ingest plain text"""
        data_id = hashlib.sha256(
            f"{text[:100]}{datetime.utcnow()}".encode()
        ).hexdigest()[:16]
        
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        return IngestedData(
            data_id=data_id,
            source_type=text_type,
            source_url=None,
            content=text,
            metadata=metadata,
            ingested_at=datetime.utcnow(),
            chunks=chunks
        )
    
    async def ingest_batch(
        self,
        sources: List[Dict[str, str]]
    ) -> List[IngestedData]:
        """Ingest multiple sources in batch"""
        logger.info(f"Batch ingestion: {len(sources)} sources")
        
        tasks = [
            self.ingest(
                source["type"],
                source["source"],
                source.get("metadata")
            )
            for source in sources
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"✅ Batch complete: {len(successful)}/{len(sources)} successful")
        
        return successful
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        by_type = {}
        for entry in self.ingestion_history:
            source_type = entry["source_type"]
            by_type[source_type] = by_type.get(source_type, 0) + 1
        
        return {
            "total_ingestions": len(self.ingestion_history),
            "by_type": by_type,
            "latest_ingestion": self.ingestion_history[-1] if self.ingestion_history else None
        }


if __name__ == "__main__":
    # Demo
    async def demo():
        print("📥 Multi-Modal Ingestion Demo\n")
        
        from grace.memory.persistent_memory import PersistentMemory
        
        memory = PersistentMemory()
        ingestion = MultiModalIngestionEngine(memory)
        
        # Ingest different types
        print("Ingesting various data sources...\n")
        
        # Web page
        web_data = await ingestion.ingest(
            "web",
            "https://example.com/article",
            {"title": "Example Article"}
        )
        print(f"✅ Web: {web_data.data_id}")
        
        # PDF
        pdf_data = await ingestion.ingest(
            "pdf",
            "./documents/paper.pdf",
            {"title": "Research Paper"}
        )
        print(f"✅ PDF: {pdf_data.data_id}")
        
        # Text
        text_data = await ingestion.ingest(
            "text",
            "This is important knowledge Grace should remember.",
            {"source": "user_input"}
        )
        print(f"✅ Text: {text_data.data_id}")
        
        # Stats
        stats = ingestion.get_ingestion_stats()
        print(f"\n📊 Ingestion Stats:")
        print(f"  Total: {stats['total_ingestions']}")
        print(f"  By type: {stats['by_type']}")
        
        # Memory stats
        mem_stats = memory.get_stats()
        print(f"\n💾 Memory Stats:")
        print(f"  Documents: {mem_stats['total_documents_ingested']}")
    
    asyncio.run(demo())

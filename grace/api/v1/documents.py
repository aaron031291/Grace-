"""
Documents and Search API endpoints
"""

from datetime import datetime, timezone
from typing import List, Optional
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from grace.auth.models import User
from grace.auth.dependencies import get_current_user
from grace.database import get_db
from grace.documents.models import Document
from grace.embeddings.service import EmbeddingService
from grace.vectorstore.service import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# Initialize services (singleton pattern)
_embedding_service = None
_vector_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_service() -> VectorStoreService:
    """Get or create vector store service"""
    global _vector_service
    if _vector_service is None:
        embedding_service = get_embedding_service()
        _vector_service = VectorStoreService(
            dimension=embedding_service.dimension,
            index_path="./data/document_vectors.bin"
        )
    return _vector_service


# Pydantic schemas
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    content_type: str = "text/plain"
    source: Optional[str] = None
    tags: List[str] = []
    metadata: Optional[dict] = {}
    is_public: bool = False


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None
    is_public: Optional[bool] = None


class DocumentResponse(BaseModel):
    id: str
    user_id: str
    title: str
    content: str
    content_type: str
    source: Optional[str]
    tags: List[str] = []
    metadata: Optional[dict]
    is_indexed: bool
    is_public: bool
    word_count: int
    view_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    id: str
    title: str
    content_preview: str
    content_type: str
    tags: List[str] = []
    is_indexed: bool
    word_count: int
    view_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=100)
    filter: Optional[dict] = None


class SearchResult(BaseModel):
    document_id: str
    title: str
    content_preview: str
    score: float
    metadata: dict
    tags: List[str] = []
    created_at: datetime


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    embedding_provider: str


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    document: DocumentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new document and index it"""
    
    # Create document
    doc_id = str(uuid.uuid4())
    word_count = len(document.content.split())
    
    db_document = Document(
        id=doc_id,
        user_id=current_user.id,
        title=document.title,
        content=document.content,
        content_type=document.content_type,
        source=document.source,
        tags=document.tags if document.tags else [],
        metadata_json=document.metadata,
        is_public=document.is_public,
        word_count=word_count
    )
    
    db.add(db_document)
    db.flush()  # Get ID without committing
    
    # Embed and index the document
    try:
        embedding_service = get_embedding_service()
        vector_service = get_vector_service()
        
        # Create text to embed (title + content)
        text_to_embed = f"{document.title}\n\n{document.content}"
        embedding = embedding_service.embed_text(text_to_embed)
        
        # Store in vector store
        metadata = {
            "document_id": doc_id,
            "user_id": current_user.id,
            "title": document.title,
            "content_preview": document.content[:200],
            "tags": document.tags,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_public": document.is_public
        }
        
        vector_ids = vector_service.get_store().add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[doc_id]
        )
        
        db_document.vector_id = vector_ids[0]
        db_document.is_indexed = True
        db_document.embedding_model = embedding_service.provider.__class__.__name__
        
        logger.info(f"Document {doc_id} indexed successfully")
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        db_document.is_indexed = False
    
    db.commit()
    db.refresh(db_document)
    
    return DocumentResponse(
        id=db_document.id,
        user_id=db_document.user_id,
        title=db_document.title,
        content=db_document.content,
        content_type=db_document.content_type,
        source=db_document.source,
        tags=db_document.tags or [],
        metadata=db_document.metadata_json,
        is_indexed=db_document.is_indexed,
        is_public=db_document.is_public,
        word_count=db_document.word_count,
        view_count=db_document.view_count,
        created_at=db_document.created_at,
        updated_at=db_document.updated_at
    )


@router.get("", response_model=List[DocumentListResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's documents with metadata"""
    
    query = db.query(Document).filter(Document.user_id == current_user.id)
    
    # Filter by tags if provided
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        # This is a simplified filter - in production, use proper JSON queries
        for tag in tag_list:
            query = query.filter(Document.tags.contains([tag]))
    
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return [
        DocumentListResponse(
            id=doc.id,
            title=doc.title,
            content_preview=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            content_type=doc.content_type,
            tags=doc.tags or [],
            is_indexed=doc.is_indexed,
            word_count=doc.word_count,
            view_count=doc.view_count,
            created_at=doc.created_at
        )
        for doc in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific document"""
    
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Update access stats
    document.view_count += 1
    document.last_accessed = datetime.now(timezone.utc)
    db.commit()
    
    return DocumentResponse(
        id=document.id,
        user_id=document.user_id,
        title=document.title,
        content=document.content,
        content_type=document.content_type,
        source=document.source,
        tags=document.tags or [],
        metadata=document.metadata_json,
        is_indexed=document.is_indexed,
        is_public=document.is_public,
        word_count=document.word_count,
        view_count=document.view_count,
        created_at=document.created_at,
        updated_at=document.updated_at
    )


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    update: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a document and re-index if content changed"""
    
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    content_changed = False
    
    # Update fields
    if update.title is not None:
        document.title = update.title
        content_changed = True
    
    if update.content is not None:
        document.content = update.content
        document.word_count = len(update.content.split())
        content_changed = True
    
    if update.tags is not None:
        document.tags = update.tags
    
    if update.metadata is not None:
        document.metadata_json = update.metadata
    
    if update.is_public is not None:
        document.is_public = update.is_public
    
    # Re-index if content changed
    if content_changed and document.vector_id:
        try:
            embedding_service = get_embedding_service()
            vector_service = get_vector_service()
            
            # Delete old vector
            vector_service.get_store().delete([document.vector_id])
            
            # Create new embedding
            text_to_embed = f"{document.title}\n\n{document.content}"
            embedding = embedding_service.embed_text(text_to_embed)
            
            # Store new vector
            metadata = {
                "document_id": document.id,
                "user_id": current_user.id,
                "title": document.title,
                "content_preview": document.content[:200],
                "tags": document.tags or [],
                "created_at": document.created_at.isoformat(),
                "is_public": document.is_public
            }
            
            vector_ids = vector_service.get_store().add_vectors(
                vectors=[embedding],
                metadata=[metadata],
                ids=[document.id]
            )
            
            document.vector_id = vector_ids[0]
            logger.info(f"Document {document_id} re-indexed")
            
        except Exception as e:
            logger.error(f"Error re-indexing document: {e}")
    
    db.commit()
    db.refresh(document)
    
    return DocumentResponse(
        id=document.id,
        user_id=document.user_id,
        title=document.title,
        content=document.content,
        content_type=document.content_type,
        source=document.source,
        tags=document.tags or [],
        metadata=document.metadata_json,
        is_indexed=document.is_indexed,
        is_public=document.is_public,
        word_count=document.word_count,
        view_count=document.view_count,
        created_at=document.created_at,
        updated_at=document.updated_at
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document and its vectors"""
    
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete from vector store
    if document.vector_id:
        try:
            vector_service = get_vector_service()
            vector_service.get_store().delete([document.vector_id])
            logger.info(f"Deleted vector for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
    
    db.delete(document)
    db.commit()
    
    logger.info(f"Deleted document {document_id}")


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Search documents using vector similarity"""
    
    try:
        embedding_service = get_embedding_service()
        vector_service = get_vector_service()
        
        # Embed the query
        query_embedding = embedding_service.embed_text(search.query)
        
        # Search vector store with user filter
        filter_dict = search.filter or {}
        filter_dict["user_id"] = current_user.id
        
        results = vector_service.get_store().search(
            query_vector=query_embedding,
            k=search.k,
            filter=filter_dict
        )
        
        # Format results
        search_results = [
            SearchResult(
                document_id=doc_id,
                title=metadata.get("title", "Untitled"),
                content_preview=metadata.get("content_preview", ""),
                score=score,
                metadata=metadata,
                tags=metadata.get("tags", []),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now(timezone.utc).isoformat()))
            )
            for doc_id, score, metadata in results
        ]
        
        return SearchResponse(
            query=search.query,
            results=search_results,
            total_results=len(search_results),
            embedding_provider=embedding_service.provider.__class__.__name__
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/system/info")
async def get_system_info(current_user: User = Depends(get_current_user)):
    """Get embedding and vector store system information"""
    
    embedding_service = get_embedding_service()
    vector_service = get_vector_service()
    
    return {
        "embedding": embedding_service.get_provider_info(),
        "vector_store": vector_service.get_store_info()
    }

"""
Grace AI Memory Service - Long-term episodic and semantic memory
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class MemoryEntry:
    """Represents a single memory entry."""
    
    def __init__(self, content: str, memory_type: str, tags: List[str] = None, importance: float = 1.0):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type  # "episodic" or "semantic"
        self.tags = tags or []
        self.importance = min(1.0, max(0.0, importance))
        self.created_at = datetime.now().isoformat()
        self.last_accessed = datetime.now().isoformat()
        self.access_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "tags": self.tags,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }

class MemoryService:
    """Manages Grace's long-term memory - episodic and semantic."""
    
    def __init__(self, storage_path: str = "memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.episodic_memory: Dict[str, MemoryEntry] = {}  # Events and experiences
        self.semantic_memory: Dict[str, MemoryEntry] = {}  # Facts and knowledge
        self.working_memory: List[str] = []  # Current active memories
        
        self._load_memory()
    
    async def store_episodic(self, content: str, tags: List[str] = None, importance: float = 1.0) -> str:
        """Store an episodic memory (event/experience)."""
        entry = MemoryEntry(content, "episodic", tags, importance)
        self.episodic_memory[entry.id] = entry
        self._save_memory()
        
        logger.info(f"Stored episodic memory: {entry.id}")
        return entry.id
    
    async def store_semantic(self, content: str, tags: List[str] = None, importance: float = 1.0) -> str:
        """Store a semantic memory (fact/knowledge)."""
        entry = MemoryEntry(content, "semantic", tags, importance)
        self.semantic_memory[entry.id] = entry
        self._save_memory()
        
        logger.info(f"Stored semantic memory: {entry.id}")
        return entry.id
    
    async def recall_episodic(self, tags: List[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Recall episodic memories, optionally filtered by tags."""
        memories = list(self.episodic_memory.values())
        
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        # Update access counts
        for memory in memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now().isoformat()
        
        self._save_memory()
        
        return memories[:limit]
    
    async def recall_semantic(self, tags: List[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Recall semantic memories, optionally filtered by tags."""
        memories = list(self.semantic_memory.values())
        
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Sort by importance
        memories.sort(key=lambda m: m.importance, reverse=True)
        
        # Update access counts
        for memory in memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now().isoformat()
        
        self._save_memory()
        
        return memories[:limit]
    
    async def search_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search through all memories for relevant content."""
        results = []
        
        for memory in list(self.episodic_memory.values()) + list(self.semantic_memory.values()):
            if query.lower() in memory.content.lower():
                results.append(memory)
        
        # Sort by importance
        results.sort(key=lambda m: m.importance, reverse=True)
        
        return results[:limit]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        total_episodic = len(self.episodic_memory)
        total_semantic = len(self.semantic_memory)
        
        avg_episodic_importance = sum(m.importance for m in self.episodic_memory.values()) / max(1, total_episodic)
        avg_semantic_importance = sum(m.importance for m in self.semantic_memory.values()) / max(1, total_semantic)
        
        return {
            "episodic_count": total_episodic,
            "semantic_count": total_semantic,
            "avg_episodic_importance": avg_episodic_importance,
            "avg_semantic_importance": avg_semantic_importance,
            "total_memories": total_episodic + total_semantic
        }
    
    def _load_memory(self):
        """Load memory from storage."""
        episodic_file = self.storage_path / "episodic.json"
        semantic_file = self.storage_path / "semantic.json"
        
        try:
            if episodic_file.exists():
                with open(episodic_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(entry_data['content'], 'episodic', entry_data.get('tags', []))
                        entry.id = entry_data['id']
                        entry.importance = entry_data.get('importance', 1.0)
                        entry.access_count = entry_data.get('access_count', 0)
                        self.episodic_memory[entry.id] = entry
            
            if semantic_file.exists():
                with open(semantic_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(entry_data['content'], 'semantic', entry_data.get('tags', []))
                        entry.id = entry_data['id']
                        entry.importance = entry_data.get('importance', 1.0)
                        entry.access_count = entry_data.get('access_count', 0)
                        self.semantic_memory[entry.id] = entry
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
    
    def _save_memory(self):
        """Save memory to storage."""
        try:
            episodic_file = self.storage_path / "episodic.json"
            semantic_file = self.storage_path / "semantic.json"
            
            with open(episodic_file, 'w') as f:
                json.dump([m.to_dict() for m in self.episodic_memory.values()], f, indent=2)
            
            with open(semantic_file, 'w') as f:
                json.dump([m.to_dict() for m in self.semantic_memory.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")

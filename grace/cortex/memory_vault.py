"""
Memory Vault - Persistent storage of system experiences (OLD Cortex enhanced)
Production implementation with timezone-aware timestamps
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
import json
import logging
import uuid
import os

logger = logging.getLogger(__name__)


class MemoryVault:
    """
    Manages persistent storage of system experiences and knowledge
    Original Cortex implementation with timezone fixes
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger("grace.cortex.memory")
        self.experiences: List[Dict[str, Any]] = []
        self.memory_lock = Lock()
        
        # Setup storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.environ.get("GRACE_DATA_PATH", "/var/lib/grace")) / "cortex_memory"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Memory vault initialized at {self.storage_path}")
        
        self._load_memories()
    
    def _load_memories(self) -> None:
        """Load memories from storage"""
        try:
            memory_files = list(self.storage_path.glob("*.json"))
            self.logger.info(f"Loading memories from {len(memory_files)} files")
            
            for memory_file in memory_files:
                try:
                    with open(memory_file, "r") as f:
                        memories = json.load(f)
                    
                    if isinstance(memories, list):
                        with self.memory_lock:
                            self.experiences.extend(memories)
                
                except Exception as e:
                    self.logger.error(f"Error loading memories from {memory_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error loading memories: {e}")
    
    def _save_memories(self) -> bool:
        """Save memories to storage"""
        try:
            with self.memory_lock:
                # Group by month
                experiences_by_month = {}
                
                for exp in self.experiences:
                    timestamp = exp.get("timestamp", "")
                    if not timestamp:
                        continue
                    
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        month_key = f"{dt.year}-{dt.month:02d}"
                    except ValueError:
                        month_key = "unknown"
                    
                    if month_key not in experiences_by_month:
                        experiences_by_month[month_key] = []
                    experiences_by_month[month_key].append(exp)
                
                # Save each month
                for month_key, month_experiences in experiences_by_month.items():
                    file_path = self.storage_path / f"memories-{month_key}.json"
                    with open(file_path, "w") as f:
                        json.dump(month_experiences, f, indent=2)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving memories: {e}")
            return False
    
    def store_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Store new experience"""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            experience_id = str(uuid.uuid4())
            
            experience_record = {
                "id": experience_id,
                "timestamp": timestamp,
                "data": experience,
                "metadata": {"storage_version": "1.0"}
            }
            
            with self.memory_lock:
                self.experiences.append(experience_record)
                
                # Save periodically
                if len(self.experiences) % 10 == 0:
                    self._save_memories()
            
            self.logger.info(f"Stored experience: {experience_id}")
            
            return experience_record
        
        except Exception as e:
            self.logger.error(f"Error storing experience: {e}")
            raise ValueError(f"Failed to store experience: {e}")
    
    def get_experience(self, experience_id: str) -> Dict[str, Any]:
        """Get experience by ID"""
        with self.memory_lock:
            for exp in self.experiences:
                if exp.get("id") == experience_id:
                    return exp.copy()
            raise ValueError(f"Experience {experience_id} not found")
    
    def search_experiences(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search experiences based on criteria"""
        try:
            results = []
            
            with self.memory_lock:
                for exp in self.experiences:
                    match = True
                    
                    for key, value in query.items():
                        if key == "time_range":
                            start_time = value.get("start")
                            end_time = value.get("end")
                            exp_time = exp.get("timestamp")
                            
                            if not exp_time:
                                match = False
                                break
                            
                            if start_time and exp_time < start_time:
                                match = False
                                break
                            
                            if end_time and exp_time > end_time:
                                match = False
                                break
                        
                        elif key == "data":
                            exp_data = exp.get("data", {})
                            for data_key, data_value in value.items():
                                if data_key not in exp_data or exp_data[data_key] != data_value:
                                    match = False
                                    break
                        
                        elif key == "text_search":
                            exp_str = json.dumps(exp).lower()
                            if value.lower() not in exp_str:
                                match = False
                                break
                    
                    if match:
                        results.append(exp.copy())
                        if len(results) >= limit:
                            break
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error searching experiences: {e}")
            raise ValueError(f"Failed to search experiences: {e}")
    
    def get_recent_experiences(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent experiences"""
        with self.memory_lock:
            sorted_experiences = sorted(
                self.experiences,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            return [exp.copy() for exp in sorted_experiences[:count]]
    
    def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience"""
        try:
            with self.memory_lock:
                for i, exp in enumerate(self.experiences):
                    if exp.get("id") == experience_id:
                        del self.experiences[i]
                        self._save_memories()
                        self.logger.info(f"Deleted experience: {experience_id}")
                        return True
                
                self.logger.warning(f"Experience not found: {experience_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error deleting experience {experience_id}: {e}")
            return False
    
    def summarize_experiences(
        self,
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate summary of experiences"""
        try:
            with self.memory_lock:
                if not self.experiences:
                    return {
                        "count": 0,
                        "time_range": {"start": None, "end": None},
                        "categories": {}
                    }
                
                filtered_experiences = self.experiences
                
                # Apply time range filter
                if time_range:
                    start_time = time_range.get("start")
                    end_time = time_range.get("end")
                    
                    if start_time or end_time:
                        filtered_experiences = []
                        for exp in self.experiences:
                            exp_time = exp.get("timestamp", "")
                            if not exp_time:
                                continue
                            if start_time and exp_time < start_time:
                                continue
                            if end_time and exp_time > end_time:
                                continue
                            filtered_experiences.append(exp)
                
                # Get time range
                timestamps = [
                    exp.get("timestamp", "") 
                    for exp in filtered_experiences 
                    if exp.get("timestamp")
                ]
                start = min(timestamps) if timestamps else None
                end = max(timestamps) if timestamps else None
                
                # Count by category
                categories = {}
                for exp in filtered_experiences:
                    exp_data = exp.get("data", {})
                    category = exp_data.get("category", "uncategorized")
                    categories[category] = categories.get(category, 0) + 1
                
                return {
                    "count": len(filtered_experiences),
                    "time_range": {"start": start, "end": end},
                    "categories": categories
                }
        
        except Exception as e:
            self.logger.error(f"Error summarizing experiences: {e}")
            return {
                "error": str(e),
                "count": 0,
                "time_range": {"start": None, "end": None},
                "categories": {}
            }

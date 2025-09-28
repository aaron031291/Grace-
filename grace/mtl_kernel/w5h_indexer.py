"""W5H Indexer - Who/What/When/Where/Why/How indexing system."""
import re
from datetime import datetime
from ..utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Set

from ..contracts.dto_common import W5HIndex, MemoryEntry


class W5HIndexer:
    """Extracts and indexes W5H elements from content."""
    
    def __init__(self):
        # Simple pattern matching for development
        self.who_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Names
            r'\b(@\w+)\b',  # Handles
            r'\b(user|admin|system|operator)\b',  # Roles
        ]
        
        self.what_patterns = [
            r'\b(create|update|delete|modify|process|analyze)\w*\b',  # Actions
            r'\b(document|file|record|entry|data)\b',  # Objects
        ]
        
        self.where_patterns = [
            r'\b(server|database|system|local|remote)\b',  # Locations
            r'\b([a-z]+://[\w\.\/\-]+)\b',  # URLs
        ]
        
        self.why_patterns = [
            r'\bbecause\s+([^.]+)\b',  # Reasons
            r'\bfor\s+([^.]+)\b',  # Purposes
        ]
        
        self.how_patterns = [
            r'\busing\s+([^.]+)\b',  # Methods
            r'\bvia\s+([^.]+)\b',  # Mechanisms
        ]
    
    def extract(self, content: str) -> W5HIndex:
        """Extract W5H elements from content."""
        index = W5HIndex()
        
        # WHO - Extract people, entities, roles
        for pattern in self.who_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.who.extend(matches)
        
        # WHAT - Extract actions, objects, subjects
        for pattern in self.what_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.what.extend(matches)
        
        # WHERE - Extract locations, systems, contexts
        for pattern in self.where_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.where.extend(matches)
        
        # WHY - Extract reasons, purposes, motivations
        for pattern in self.why_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.why.extend(matches)
        
        # HOW - Extract methods, mechanisms, processes
        for pattern in self.how_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.how.extend(matches)
        
        # WHEN - Set current timestamp (could be enhanced with NLP)
        index.when = utc_now()
        
        # Deduplicate and clean up
        index.who = list(set(index.who))
        index.what = list(set(index.what))
        index.where = list(set(index.where))
        index.why = list(set(index.why))
        index.how = list(set(index.how))
        
        return index
    
    def enhance_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Enhance memory entry with W5H indexing."""
        if not entry.w5h_index or not any([
            entry.w5h_index.who, entry.w5h_index.what, entry.w5h_index.where,
            entry.w5h_index.why, entry.w5h_index.how
        ]):
            entry.w5h_index = self.extract(entry.content)
        
        return entry
    
    def search_by_w5h(self, entries: List[MemoryEntry], 
                      who: Optional[str] = None,
                      what: Optional[str] = None, 
                      where: Optional[str] = None,
                      why: Optional[str] = None,
                      how: Optional[str] = None) -> List[MemoryEntry]:
        """Search entries by W5H criteria."""
        results = []
        
        for entry in entries:
            match = True
            
            if who and not any(who.lower() in person.lower() for person in entry.w5h_index.who):
                match = False
            
            if what and not any(what.lower() in thing.lower() for thing in entry.w5h_index.what):
                match = False
            
            if where and not any(where.lower() in place.lower() for place in entry.w5h_index.where):
                match = False
            
            if why and not any(why.lower() in reason.lower() for reason in entry.w5h_index.why):
                match = False
            
            if how and not any(how.lower() in method.lower() for method in entry.w5h_index.how):
                match = False
            
            if match:
                results.append(entry)
        
        return results
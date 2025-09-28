"""Active learning query strategies for optimal data selection."""

import json
import random
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


def select_batch(uncertainty: List[float], diversity: List[float], coresets: List[int], cfg: Dict[str, Any]) -> List[str]:
    """Select batch of items for active learning based on strategy configuration."""
    strategy = cfg.get("strategy", "uncertainty")
    batch_size = cfg.get("batch_size", 32)
    min_confidence = cfg.get("min_confidence", 0.0)
    
    # Simple implementation - in production would use proper ML models
    if strategy == "uncertainty":
        return _uncertainty_sampling(uncertainty, batch_size, min_confidence)
    elif strategy == "margin":
        return _margin_sampling(uncertainty, batch_size, min_confidence)
    elif strategy == "entropy":
        return _entropy_sampling(uncertainty, batch_size, min_confidence)
    elif strategy == "diversity":
        return _diversity_sampling(diversity, batch_size)
    elif strategy == "coresets":
        return _coreset_sampling(coresets, batch_size)
    elif strategy == "hybrid":
        return _hybrid_sampling(uncertainty, diversity, batch_size, min_confidence)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _uncertainty_sampling(uncertainty: List[float], batch_size: int, min_confidence: float = 0.0) -> List[str]:
    """Select items with highest prediction uncertainty."""
    # Filter by min confidence and sort by uncertainty (descending)
    candidates = [(i, u) for i, u in enumerate(uncertainty) if (1.0 - u) >= min_confidence]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Select top batch_size items
    selected = [str(idx) for idx, _ in candidates[:batch_size]]
    return selected


def _margin_sampling(uncertainty: List[float], batch_size: int, min_confidence: float = 0.0) -> List[str]:
    """Select items with smallest margin between top predictions."""
    # Simulate margin calculation - in practice would use actual prediction margins
    margins = [abs(u - 0.5) for u in uncertainty]  # Distance from decision boundary
    candidates = [(i, m) for i, m in enumerate(margins) if (1.0 - uncertainty[i]) >= min_confidence]
    candidates.sort(key=lambda x: x[1])  # Sort by smallest margin
    
    selected = [str(idx) for idx, _ in candidates[:batch_size]]
    return selected


def _entropy_sampling(uncertainty: List[float], batch_size: int, min_confidence: float = 0.0) -> List[str]:
    """Select items with highest prediction entropy."""
    # Simulate entropy calculation
    entropies = [-u * np.log2(u + 1e-10) - (1-u) * np.log2(1-u + 1e-10) for u in uncertainty]
    candidates = [(i, e) for i, e in enumerate(entropies) if (1.0 - uncertainty[i]) >= min_confidence]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [str(idx) for idx, _ in candidates[:batch_size]]
    return selected


def _diversity_sampling(diversity: List[float], batch_size: int) -> List[str]:
    """Select diverse items to maximize coverage."""
    # Simple diversity sampling - in practice would use embeddings and clustering
    candidates = list(enumerate(diversity))
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    selected = [str(idx) for idx, _ in candidates[:batch_size]]
    return selected


def _coreset_sampling(coresets: List[int], batch_size: int) -> List[str]:
    """Select representative core set of items."""
    # Use provided coreset indices
    selected = [str(idx) for idx in coresets[:batch_size]]
    return selected


def _hybrid_sampling(uncertainty: List[float], diversity: List[float], batch_size: int, min_confidence: float = 0.0) -> List[str]:
    """Combine uncertainty and diversity for balanced selection."""
    # Weight uncertainty and diversity scores
    uncertainty_weight = 0.7
    diversity_weight = 0.3
    
    scores = []
    for i, (u, d) in enumerate(zip(uncertainty, diversity)):
        if (1.0 - u) >= min_confidence:
            combined_score = uncertainty_weight * u + diversity_weight * d
            scores.append((i, combined_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [str(idx) for idx, _ in scores[:batch_size]]
    return selected


class ActiveLearningStrategies:
    """Manages active learning query strategies and batch selection."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def select_batch(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Select batch of items for labeling using active learning."""
        dataset_id = cfg.get("dataset_id")
        version = cfg.get("version")
        strategy = cfg.get("strategy", "uncertainty")
        batch_size = cfg.get("batch_size", 32)
        
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        # Generate query ID
        query_id = f"aq_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Simulate getting unlabeled data and model predictions
        # In practice, this would integrate with actual ML models and data storage
        unlabeled_items = self._get_unlabeled_items(dataset_id, version)
        
        if not unlabeled_items:
            return {"items": [], "message": "No unlabeled items available"}
        
        # Generate mock uncertainty/diversity scores
        uncertainty_scores = [random.uniform(0.3, 0.9) for _ in unlabeled_items]
        diversity_scores = [random.uniform(0.1, 0.8) for _ in unlabeled_items]
        coreset_indices = list(range(min(len(unlabeled_items), batch_size * 2)))
        
        # Select batch using strategy
        selected_indices = select_batch(uncertainty_scores, diversity_scores, coreset_indices, cfg)
        selected_items = [unlabeled_items[int(idx)] for idx in selected_indices if int(idx) < len(unlabeled_items)]
        
        # Store query in database
        self._store_query(query_id, cfg, selected_items)
        
        # Calculate estimated query gain (mock)
        query_gain_f1 = self._estimate_query_gain(strategy, len(selected_items))
        
        return {
            "query_id": query_id,
            "items": selected_items[:batch_size],
            "strategy": strategy,
            "estimated_gain_f1": query_gain_f1,
            "total_unlabeled": len(unlabeled_items)
        }
    
    def _get_unlabeled_items(self, dataset_id: str, version: Optional[str]) -> List[str]:
        """Get list of unlabeled items for a dataset."""
        # Mock implementation - return sample item IDs
        # In practice, would query actual data storage
        base_count = random.randint(100, 1000)
        return [f"item_{dataset_id}_{i}" for i in range(base_count)]
    
    def _store_query(self, query_id: str, cfg: Dict[str, Any], selected_items: List[str]):
        """Store active learning query in database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO active_queries (
                    query_id, dataset_id, version, strategy, batch_size,
                    segment_filters_json, min_confidence, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                cfg.get("dataset_id"),
                cfg.get("version"),
                cfg.get("strategy"),
                cfg.get("batch_size"),
                json.dumps(cfg.get("segment_filters", {})),
                cfg.get("min_confidence"),
                json.dumps(selected_items)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _estimate_query_gain(self, strategy: str, batch_size: int) -> float:
        """Estimate potential F1 gain from active learning query."""
        # Mock estimation - in practice would use historical performance data
        base_gain = {
            "uncertainty": 0.15,
            "margin": 0.12,
            "entropy": 0.14,
            "diversity": 0.08,
            "coresets": 0.10,
            "hybrid": 0.18
        }.get(strategy, 0.10)
        
        # Scale by batch size (diminishing returns)
        size_factor = min(1.0, batch_size / 50.0)
        estimated_gain = base_gain * size_factor
        
        return round(estimated_gain, 3)
    
    def get_query_history(self, dataset_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get active learning query history for a dataset."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query_id, version, strategy, batch_size, query_gain_f1, created_at
                FROM active_queries
                WHERE dataset_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (dataset_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "query_id": row["query_id"],
                    "version": row["version"],
                    "strategy": row["strategy"],
                    "batch_size": row["batch_size"],
                    "query_gain_f1": row["query_gain_f1"],
                    "created_at": row["created_at"]
                })
            
            return history
        finally:
            conn.close()
    
    def update_query_results(self, query_id: str, actual_gain_f1: float):
        """Update query results with actual performance gain."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE active_queries
                SET query_gain_f1 = ?
                WHERE query_id = ?
            """, (actual_gain_f1, query_id))
            conn.commit()
        finally:
            conn.close()
    
    def get_strategy_performance(self, dataset_id: str) -> Dict[str, Any]:
        """Get performance statistics for different active learning strategies."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT strategy, 
                       COUNT(*) as query_count,
                       AVG(query_gain_f1) as avg_gain,
                       AVG(batch_size) as avg_batch_size
                FROM active_queries
                WHERE dataset_id = ? AND query_gain_f1 IS NOT NULL
                GROUP BY strategy
                ORDER BY avg_gain DESC
            """, (dataset_id,))
            
            performance = {}
            for row in cursor.fetchall():
                performance[row["strategy"]] = {
                    "query_count": row["query_count"],
                    "avg_gain_f1": round(row["avg_gain"], 3),
                    "avg_batch_size": round(row["avg_batch_size"], 1)
                }
            
            return performance
        finally:
            conn.close()
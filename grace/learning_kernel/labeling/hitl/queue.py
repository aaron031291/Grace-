"""Human-in-the-loop labeling queue and task management."""

import json
import sqlite3
import random
from datetime import datetime
from typing import Dict, List, Optional, Any


class HITLQueue:
    """Manages human-in-the-loop labeling tasks and queues."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def create_task(self, task: Dict[str, Any]) -> str:
        """Create a new labeling task."""
        # Validate required fields
        required_fields = ["task_id", "dataset_id", "version", "policy_id", "items"]
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate task_id pattern
        task_id = task["task_id"]
        if not task_id.startswith("lab_") or len(task_id) < 10:
            raise ValueError("task_id must match pattern 'lab_[a-z0-9]{6,}'")
        
        # Validate priority and assignment strategy
        valid_priorities = ["low", "normal", "high", "critical"]
        valid_assignments = ["auto", "round_robin", "skill_based"]
        
        priority = task.get("priority", "normal")
        assign_strategy = task.get("assign", "auto")
        
        if priority not in valid_priorities:
            raise ValueError(f"Invalid priority. Must be one of: {valid_priorities}")
        
        if assign_strategy not in valid_assignments:
            raise ValueError(f"Invalid assignment strategy. Must be one of: {valid_assignments}")
        
        # Store task in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO label_tasks (
                    task_id, dataset_id, version, policy_id, assign_strategy,
                    items_json, priority, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                task["dataset_id"],
                task["version"],
                task["policy_id"],
                assign_strategy,
                json.dumps(task["items"]),
                priority,
                "active"
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Task {task_id} already exists")
        finally:
            conn.close()
        
        return task_id
    
    def submit_label(self, label: Dict[str, Any]) -> str:
        """Submit a label for a task item."""
        # Validate required fields
        required_fields = ["label_id", "task_id", "item_id", "annotator_id", "y", "quality", "at"]
        for field in required_fields:
            if field not in label:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate task exists
        task = self.get_task(label["task_id"])
        if not task:
            raise ValueError(f"Task {label['task_id']} not found")
        
        # Store label in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO labels (
                    label_id, task_id, item_id, annotator_id, y_value,
                    evidence_json, weak_sources_json, agreement, gold_correct, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                label["label_id"],
                label["task_id"],
                label["item_id"],
                label["annotator_id"],
                json.dumps(label["y"]),
                json.dumps(label.get("evidence", {})),
                json.dumps(label.get("weak_sources", [])),
                label["quality"].get("agreement"),
                label["quality"].get("gold_correct"),
                label.get("confidence", 1.0)
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Label {label['label_id']} already exists")
        finally:
            conn.close()
        
        return label["label_id"]
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM label_tasks WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
                "version": row["version"],
                "policy_id": row["policy_id"],
                "assign_strategy": row["assign_strategy"],
                "items": json.loads(row["items_json"]),
                "priority": row["priority"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        finally:
            conn.close()
    
    def get_queue_for_annotator(self, annotator_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get labeling queue for a specific annotator."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            # Get tasks assigned to this annotator or available for assignment
            cursor.execute("""
                SELECT lt.*, 
                       COUNT(l.label_id) as labeled_count,
                       json_array_length(lt.items_json) as total_items
                FROM label_tasks lt
                LEFT JOIN labels l ON lt.task_id = l.task_id
                WHERE lt.status = 'active'
                AND (
                    lt.assign_strategy = 'auto'
                    OR NOT EXISTS (
                        SELECT 1 FROM labels l2 
                        WHERE l2.task_id = lt.task_id 
                        AND l2.annotator_id != ?
                    )
                )
                GROUP BY lt.task_id
                HAVING labeled_count < total_items
                ORDER BY 
                    CASE lt.priority
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'normal' THEN 3
                        WHEN 'low' THEN 4
                    END,
                    lt.created_at ASC
                LIMIT ?
            """, (annotator_id, limit))
            
            tasks = []
            for row in cursor.fetchall():
                # Get unlabeled items for this task
                items = json.loads(row["items_json"])
                labeled_items = self._get_labeled_items(row["task_id"], annotator_id)
                unlabeled_items = [item for item in items if item not in labeled_items]
                
                if unlabeled_items:
                    tasks.append({
                        "task_id": row["task_id"],
                        "dataset_id": row["dataset_id"],
                        "version": row["version"],
                        "priority": row["priority"],
                        "unlabeled_items": unlabeled_items[:5],  # Limit items per task
                        "progress": {
                            "labeled": row["labeled_count"],
                            "total": row["total_items"],
                            "remaining": len(unlabeled_items)
                        }
                    })
            
            return tasks
        finally:
            conn.close()
    
    def _get_labeled_items(self, task_id: str, annotator_id: Optional[str] = None) -> List[str]:
        """Get list of already labeled items for a task."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            if annotator_id:
                cursor.execute("""
                    SELECT item_id FROM labels 
                    WHERE task_id = ? AND annotator_id = ?
                """, (task_id, annotator_id))
            else:
                cursor.execute("""
                    SELECT DISTINCT item_id FROM labels WHERE task_id = ?
                """, (task_id,))
            
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """Get progress statistics for a labeling task."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT lt.task_id, lt.items_json, lt.priority, lt.status,
                       COUNT(l.label_id) as labeled_count,
                       AVG(l.agreement) as avg_agreement,
                       AVG(l.confidence) as avg_confidence,
                       COUNT(DISTINCT l.annotator_id) as annotator_count
                FROM label_tasks lt
                LEFT JOIN labels l ON lt.task_id = l.task_id
                WHERE lt.task_id = ?
                GROUP BY lt.task_id
            """, (task_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            total_items = len(json.loads(row["items_json"]))
            labeled_count = row["labeled_count"] or 0
            
            return {
                "task_id": task_id,
                "status": row["status"],
                "priority": row["priority"],
                "progress": {
                    "labeled": labeled_count,
                    "total": total_items,
                    "completion_rate": labeled_count / total_items if total_items > 0 else 0.0
                },
                "quality": {
                    "avg_agreement": round(row["avg_agreement"] or 0.0, 3),
                    "avg_confidence": round(row["avg_confidence"] or 0.0, 3),
                    "annotator_count": row["annotator_count"] or 0
                }
            }
        finally:
            conn.close()
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE label_tasks 
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
            """, (task_id,))
            
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_annotator_stats(self, annotator_id: str, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for an annotator."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as label_count,
                       AVG(agreement) as avg_agreement,
                       AVG(confidence) as avg_confidence,
                       COUNT(CASE WHEN gold_correct = 1 THEN 1 END) as gold_correct_count,
                       COUNT(CASE WHEN gold_correct IS NOT NULL THEN 1 END) as gold_total_count
                FROM labels
                WHERE annotator_id = ?
                AND created_at >= datetime('now', '-{} days')
            """.format(days), (annotator_id,))
            
            row = cursor.fetchone()
            
            # Calculate throughput (labels per hour)
            cursor.execute("""
                SELECT COUNT(*) as recent_labels
                FROM labels
                WHERE annotator_id = ?
                AND created_at >= datetime('now', '-1 hour')
            """, (annotator_id,))
            
            recent_count = cursor.fetchone()["recent_labels"]
            
            gold_accuracy = 0.0
            if row["gold_total_count"] > 0:
                gold_accuracy = row["gold_correct_count"] / row["gold_total_count"]
            
            return {
                "annotator_id": annotator_id,
                "period_days": days,
                "total_labels": row["label_count"] or 0,
                "avg_agreement": round(row["avg_agreement"] or 0.0, 3),
                "avg_confidence": round(row["avg_confidence"] or 0.0, 3),
                "gold_accuracy": round(gold_accuracy, 3),
                "throughput_labels_per_hour": recent_count,
                "quality_score": round((row["avg_agreement"] or 0.0) * gold_accuracy, 3)
            }
        finally:
            conn.close()
    
    def reassign_task(self, task_id: str, new_annotator_id: str) -> bool:
        """Reassign a task to a different annotator."""
        # This is a simplified implementation
        # In practice, would handle more complex reassignment logic
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE label_tasks 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
            """, (task_id,))
            
            # Note: In a full implementation, would track task assignments separately
            return cursor.rowcount > 0
        finally:
            conn.close()
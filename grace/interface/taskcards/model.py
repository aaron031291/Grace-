"""TaskCard model and lifecycle management."""
from datetime import datetime
from ...utils.datetime_utils import utc_now, iso_format, format_for_filename
from typing import Dict, List, Optional, Literal
import uuid
import logging

from ..models import TaskCard, ThreadMessage, MessageContent, TaskMetrics, generate_card_id

logger = logging.getLogger(__name__)


class TaskCardManager:
    """Manages TaskCard lifecycle and state."""
    
    def __init__(self):
        self.cards: Dict[str, TaskCard] = {}
    
    def new_card(self, title: str, kind: str, owner: str, ctx: Optional[Dict] = None) -> Dict:
        """Create a new TaskCard."""
        try:
            card_id = generate_card_id()
            
            card = TaskCard(
                card_id=card_id,
                title=title,
                kind=kind,
                owner=owner,
                state="open",
                context=ctx or {},
                created_at=utc_now(),
                thread=[],
                metrics=TaskMetrics(),
                approvals=[]
            )
            
            self.cards[card_id] = card
            logger.info(f"Created TaskCard {card_id}: {title}")
            
            return card.dict()
            
        except Exception as e:
            logger.error(f"Failed to create TaskCard: {e}")
            raise
    
    def get_card(self, card_id: str) -> Optional[TaskCard]:
        """Get TaskCard by ID."""
        return self.cards.get(card_id)
    
    def append_thread(self, card_id: str, message: Dict) -> Dict:
        """Add message to TaskCard thread."""
        if card_id not in self.cards:
            raise ValueError(f"TaskCard {card_id} not found")
        
        card = self.cards[card_id]
        
        # Create thread message
        thread_msg = ThreadMessage(
            msg_id=message.get("msg_id", str(uuid.uuid4())),
            role=message["role"],
            author=message["author"],
            at=utc_now(),
            content=MessageContent(**message.get("content", {}))
        )
        
        if card.thread is None:
            card.thread = []
        
        card.thread.append(thread_msg)
        
        logger.info(f"Added message to TaskCard {card_id} thread")
        return card.dict()
    
    def set_state(self, card_id: str, state: str) -> Dict:
        """Update TaskCard state."""
        if card_id not in self.cards:
            raise ValueError(f"TaskCard {card_id} not found")
        
        valid_states = ["open", "running", "paused", "waiting_approval", "done", "failed"]
        if state not in valid_states:
            raise ValueError(f"Invalid state: {state}. Must be one of {valid_states}")
        
        card = self.cards[card_id]
        old_state = card.state
        card.state = state
        
        logger.info(f"TaskCard {card_id} state changed: {old_state} -> {state}")
        return card.dict()
    
    def update_context(self, card_id: str, context_updates: Dict) -> Dict:
        """Update TaskCard context."""
        if card_id not in self.cards:
            raise ValueError(f"TaskCard {card_id} not found")
        
        card = self.cards[card_id]
        if card.context is None:
            card.context = {}
        
        card.context.update(context_updates)
        
        logger.info(f"Updated TaskCard {card_id} context")
        return card.dict()
    
    def update_metrics(self, card_id: str, metrics: Dict) -> Dict:
        """Update TaskCard metrics."""
        if card_id not in self.cards:
            raise ValueError(f"TaskCard {card_id} not found")
        
        card = self.cards[card_id]
        if card.metrics is None:
            card.metrics = TaskMetrics()
        
        if "latency_ms" in metrics:
            card.metrics.latency_ms = metrics["latency_ms"]
        if "steps" in metrics:
            card.metrics.steps = metrics["steps"]
        
        logger.info(f"Updated TaskCard {card_id} metrics")
        return card.dict()
    
    def add_approval(self, card_id: str, decision_id: str) -> Dict:
        """Add approval decision ID to TaskCard."""
        if card_id not in self.cards:
            raise ValueError(f"TaskCard {card_id} not found")
        
        card = self.cards[card_id]
        if card.approvals is None:
            card.approvals = []
        
        if decision_id not in card.approvals:
            card.approvals.append(decision_id)
        
        logger.info(f"Added approval {decision_id} to TaskCard {card_id}")
        return card.dict()
    
    def list_cards(self, owner: Optional[str] = None, state: Optional[str] = None, kind: Optional[str] = None) -> List[TaskCard]:
        """List TaskCards with optional filtering."""
        cards = list(self.cards.values())
        
        if owner:
            cards = [c for c in cards if c.owner == owner]
        
        if state:
            cards = [c for c in cards if c.state == state]
        
        if kind:
            cards = [c for c in cards if c.kind == kind]
        
        # Sort by created_at descending
        cards.sort(key=lambda c: c.created_at, reverse=True)
        
        return cards
    
    def delete_card(self, card_id: str) -> bool:
        """Delete a TaskCard."""
        if card_id in self.cards:
            card = self.cards.pop(card_id)
            logger.info(f"Deleted TaskCard {card_id}: {card.title}")
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get TaskCard manager statistics."""
        cards = list(self.cards.values())
        
        # Count by state
        state_counts = {}
        for card in cards:
            state_counts[card.state] = state_counts.get(card.state, 0) + 1
        
        # Count by kind
        kind_counts = {}
        for card in cards:
            kind_counts[card.kind] = kind_counts.get(card.kind, 0) + 1
        
        # Calculate average thread length
        thread_lengths = [len(card.thread) if card.thread else 0 for card in cards]
        avg_thread_length = sum(thread_lengths) / len(thread_lengths) if thread_lengths else 0
        
        return {
            "total_cards": len(cards),
            "state_distribution": state_counts,
            "kind_distribution": kind_counts,
            "avg_thread_length": avg_thread_length
        }


# Convenience functions matching the interface in the problem statement
def new_card(title: str, kind: str, owner: str, ctx: Dict) -> Dict:
    """Create a new TaskCard (convenience function)."""
    manager = TaskCardManager()
    return manager.new_card(title, kind, owner, ctx)


def append_thread(card_id: str, message: Dict) -> Dict:
    """Append message to TaskCard thread (convenience function)."""
    manager = TaskCardManager()
    return manager.append_thread(card_id, message)


def set_state(card_id: str, state: str) -> Dict:
    """Set TaskCard state (convenience function)."""
    manager = TaskCardManager()
    return manager.set_state(card_id, state)
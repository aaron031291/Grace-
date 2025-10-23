"""
Grace AI Notification Service - Sends approval requests for self-improvements
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotificationService:
    """Sends notifications requesting human approval for Grace's actions."""
    
    async def request_approval(
        self,
        proposed_change: str,
        details: Dict[str, Any],
        correlation_id: str
    ):
        """Request approval for a proposed change."""
        
        print("\n" + "="*60)
        print(">>> APPROVAL REQUESTED <<<")
        print("-"*60)
        print(f"Proposed Change:\n{proposed_change}\n")
        print("Details:")
        for key, value in details.items():
            print(f"  - {key}: {value}")
        print(f"\nCorrelation ID: {correlation_id}")
        print("-"*60)
        print("Please review and respond with 'approve' or 'reject':")
        print("="*60 + "\n")
        
        logger.info(f"Approval requested for change: {proposed_change} (ID: {correlation_id})")

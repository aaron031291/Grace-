"""Consent and autonomy flow management."""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import uuid

from ..models import ConsentRecord

logger = logging.getLogger(__name__)


class ConsentService:
    """Manages user consent for autonomy and data use."""
    
    def __init__(self):
        self.consents: Dict[str, ConsentRecord] = {}
    
    def grant_consent(self, user_id: str, scope: str, expires_days: Optional[int] = None, evidence_uri: Optional[str] = None) -> str:
        """Grant consent for a specific scope."""
        consent_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            scope=scope,
            status="granted",
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            evidence_uri=evidence_uri
        )
        
        self.consents[consent_id] = consent
        logger.info(f"Granted {scope} consent for user {user_id}")
        
        return consent_id
    
    def revoke_consent(self, consent_id: str) -> bool:
        """Revoke consent by ID."""
        if consent_id not in self.consents:
            return False
        
        consent = self.consents[consent_id]
        consent.status = "revoked"
        
        logger.info(f"Revoked {consent.scope} consent {consent_id} for user {consent.user_id}")
        return True
    
    def deny_consent(self, user_id: str, scope: str, evidence_uri: Optional[str] = None) -> str:
        """Explicitly deny consent for a scope."""
        consent_id = str(uuid.uuid4())
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            scope=scope,
            status="denied",
            created_at=datetime.utcnow(),
            evidence_uri=evidence_uri
        )
        
        self.consents[consent_id] = consent
        logger.info(f"Denied {scope} consent for user {user_id}")
        
        return consent_id
    
    def check_consent(self, user_id: str, scope: str) -> bool:
        """Check if user has valid consent for scope."""
        current_time = datetime.utcnow()
        
        user_consents = [
            c for c in self.consents.values()
            if c.user_id == user_id and c.scope == scope
        ]
        
        # Find most recent consent for this scope
        if not user_consents:
            return False
        
        latest_consent = max(user_consents, key=lambda c: c.created_at)
        
        # Check if consent is granted and not expired
        if latest_consent.status != "granted":
            return False
        
        if latest_consent.expires_at and current_time > latest_consent.expires_at:
            return False
        
        return True
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user."""
        return [
            c for c in self.consents.values()
            if c.user_id == user_id
        ]
    
    def get_consent(self, consent_id: str) -> Optional[ConsentRecord]:
        """Get consent record by ID."""
        return self.consents.get(consent_id)
    
    def list_pending_consents(self) -> List[ConsentRecord]:
        """List all pending consent requests."""
        return [
            c for c in self.consents.values()
            if c.status == "pending"
        ]
    
    def get_expiring_consents(self, days_ahead: int = 30) -> List[ConsentRecord]:
        """Get consents expiring within specified days."""
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        return [
            c for c in self.consents.values()
            if c.status == "granted" 
            and c.expires_at 
            and c.expires_at <= cutoff_date
        ]
    
    def cleanup_expired_consents(self) -> int:
        """Mark expired consents as revoked and return count."""
        current_time = datetime.utcnow()
        expired_count = 0
        
        for consent in self.consents.values():
            if (consent.status == "granted" and 
                consent.expires_at and 
                current_time > consent.expires_at):
                consent.status = "revoked"
                expired_count += 1
                logger.info(f"Marked expired consent {consent.consent_id} as revoked")
        
        return expired_count
    
    def get_consent_summary(self, user_id: str) -> Dict:
        """Get consent summary for a user."""
        user_consents = self.get_user_consents(user_id)
        
        summary = {}
        for scope in ["autonomy", "pii_use", "external_share", "canary_participation"]:
            scope_consents = [c for c in user_consents if c.scope == scope]
            
            if scope_consents:
                latest = max(scope_consents, key=lambda c: c.created_at)
                summary[scope] = {
                    "status": latest.status,
                    "granted_at": latest.created_at.isoformat() if latest.status == "granted" else None,
                    "expires_at": latest.expires_at.isoformat() if latest.expires_at else None,
                    "consent_id": latest.consent_id
                }
            else:
                summary[scope] = {
                    "status": "not_requested",
                    "granted_at": None,
                    "expires_at": None,
                    "consent_id": None
                }
        
        return summary
    
    def request_consent_renewal(self, user_id: str, scope: str) -> str:
        """Request consent renewal for expiring consent."""
        consent_id = str(uuid.uuid4())
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            scope=scope,
            status="pending",
            created_at=datetime.utcnow()
        )
        
        self.consents[consent_id] = consent
        logger.info(f"Requested {scope} consent renewal for user {user_id}")
        
        return consent_id
    
    def get_stats(self) -> Dict:
        """Get consent service statistics."""
        consents = list(self.consents.values())
        
        # Count by status
        status_counts = {}
        for consent in consents:
            status_counts[consent.status] = status_counts.get(consent.status, 0) + 1
        
        # Count by scope
        scope_counts = {}
        for consent in consents:
            scope_counts[consent.scope] = scope_counts.get(consent.scope, 0) + 1
        
        # Count expiring soon
        expiring_soon = len(self.get_expiring_consents())
        
        return {
            "total_consents": len(consents),
            "status_distribution": status_counts,
            "scope_distribution": scope_counts,
            "expiring_within_30_days": expiring_soon
        }
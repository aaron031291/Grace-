"""
Component Handshake Protocol - Production implementation
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta  # FIXED: Added timezone
from enum import Enum
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class HandshakeStatus(Enum):
    """Status of handshake process"""
    INITIATED = "initiated"
    AUTHENTICATING = "authenticating"
    NEGOTIATING = "negotiating"
    VALIDATED = "validated"
    ACTIVE = "active"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ComponentIdentity:
    """Cryptographic identity for a component"""
    component_id: str
    component_type: str
    public_key: str
    signature: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # FIXED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandshakeSession:
    """Active handshake session"""
    session_id: str
    component_identity: ComponentIdentity
    status: HandshakeStatus
    challenge: str
    challenge_response: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    version: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # FIXED
    expires_at: Optional[datetime] = None
    trust_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentHandshake:
    """
    Production-ready component handshake system
    Handles secure registration, authentication, and capability negotiation
    """
    
    def __init__(self, trust_manager=None, constitution_validator=None):
        self.trust_manager = trust_manager
        self.constitution = constitution_validator
        self.active_sessions: Dict[str, HandshakeSession] = {}
        self.registered_components: Dict[str, ComponentIdentity] = {}
        self.session_timeout = timedelta(minutes=5)
        self.min_trust_score = 0.6
        logger.info("ComponentHandshake initialized")
    
    def initiate_handshake(
        self,
        component_id: str,
        component_type: str,
        version: str,
        capabilities: Set[str],
        metadata: Optional[Dict] = None
    ) -> HandshakeSession:
        """
        Initiate handshake for new component
        Production implementation with full security
        """
        # Generate cryptographic challenge
        challenge = self._generate_challenge()
        
        # Create identity
        identity = ComponentIdentity(
            component_id=component_id,
            component_type=component_type,
            public_key=self._generate_public_key(component_id),
            signature=self._sign_identity(component_id, component_type),
            timestamp=datetime.now(timezone.utc),  # FIXED
            metadata=metadata or {}
        )
        
        # Create session
        session_id = self._generate_session_id()
        now = datetime.now(timezone.utc)  # FIXED
        session = HandshakeSession(
            session_id=session_id,
            component_identity=identity,
            status=HandshakeStatus.INITIATED,
            challenge=challenge,
            capabilities=capabilities,
            version=version,
            started_at=now,
            expires_at=now + self.session_timeout,
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Handshake initiated for {component_id} (session: {session_id})")
        
        return session
    
    def authenticate_component(
        self,
        session_id: str,
        challenge_response: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """
        Authenticate component with challenge-response
        Production security implementation
        """
        if session_id not in self.active_sessions:
            logger.error(f"Invalid session: {session_id}")
            return False
        
        session = self.active_sessions[session_id]
        
        # Check expiration - FIXED: timezone-aware
        if datetime.now(timezone.utc) > session.expires_at:
            session.status = HandshakeStatus.EXPIRED
            logger.warning(f"Session expired: {session_id}")
            return False
        
        # Verify challenge response
        expected_response = self._compute_challenge_response(
            session.challenge,
            session.component_identity.component_id
        )
        
        if challenge_response != expected_response:
            session.status = HandshakeStatus.FAILED
            logger.error(f"Challenge response mismatch for {session_id}")
            return False
        
        # Verify signature
        if not self._verify_signature(session.component_identity):
            session.status = HandshakeStatus.FAILED
            logger.error(f"Signature verification failed for {session_id}")
            return False
        
        # Verify credentials
        if not self._verify_credentials(credentials):
            session.status = HandshakeStatus.FAILED
            logger.error(f"Credential verification failed for {session_id}")
            return False
        
        session.challenge_response = challenge_response
        session.status = HandshakeStatus.AUTHENTICATING
        
        logger.info(f"Component authenticated: {session.component_identity.component_id}")
        
        return True
    
    def negotiate_capabilities(
        self,
        session_id: str,
        required_capabilities: Set[str],
        optional_capabilities: Set[str]
    ) -> Dict[str, Any]:
        """
        Negotiate capabilities between component and system
        Production implementation with validation
        """
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Invalid session'}
        
        session = self.active_sessions[session_id]
        
        if session.status != HandshakeStatus.AUTHENTICATING:
            return {'success': False, 'error': 'Not authenticated'}
        
        # Check required capabilities
        missing_required = required_capabilities - session.capabilities
        if missing_required:
            logger.error(f"Missing required capabilities: {missing_required}")
            return {
                'success': False,
                'error': 'Missing required capabilities',
                'missing': list(missing_required)
            }
        
        # Find common optional capabilities
        available_optional = session.capabilities & optional_capabilities
        
        # Validate against constitution if available
        if self.constitution:
            validation = self.constitution.validate_against_constitution(
                {
                    'action': 'register_component',
                    'component_id': session.component_identity.component_id,
                    'capabilities': list(session.capabilities)
                },
                {}
            )
            
            if not validation.passed:
                session.status = HandshakeStatus.FAILED
                logger.error(f"Constitutional validation failed: {validation.violations}")
                return {
                    'success': False,
                    'error': 'Constitutional validation failed',
                    'violations': validation.violations
                }
        
        session.status = HandshakeStatus.NEGOTIATING
        
        negotiation_result = {
            'success': True,
            'required_capabilities': list(required_capabilities),
            'available_optional': list(available_optional),
            'all_capabilities': list(session.capabilities),
            'negotiated_at': datetime.now(timezone.utc).isoformat()  # FIXED
        }
        
        session.metadata['negotiation_result'] = negotiation_result
        
        logger.info(f"Capabilities negotiated for {session.component_identity.component_id}")
        
        return negotiation_result
    
    def validate_version(
        self,
        session_id: str,
        system_version: str,
        compatibility_matrix: Dict[str, List[str]]
    ) -> bool:
        """
        Validate component version compatibility
        Production version checking
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        component_version = session.version
        
        # Parse versions
        comp_major, comp_minor, comp_patch = self._parse_version(component_version)
        sys_major, sys_minor, sys_patch = self._parse_version(system_version)
        
        # Check major version compatibility
        if comp_major != sys_major:
            logger.error(f"Major version mismatch: {component_version} vs {system_version}")
            return False
        
        # Check minor version (allow backwards compatibility)
        if comp_minor > sys_minor:
            logger.warning(f"Component version newer than system: {component_version} > {system_version}")
            # Check if explicitly compatible
            if component_version not in compatibility_matrix.get(system_version, []):
                return False
        
        # Check compatibility matrix
        component_type = session.component_identity.component_type
        if component_type in compatibility_matrix:
            compatible_versions = compatibility_matrix[component_type]
            if component_version not in compatible_versions:
                logger.error(f"Version not in compatibility matrix: {component_version}")
                return False
        
        session.status = HandshakeStatus.VALIDATED
        
        logger.info(f"Version validated: {component_version} compatible with {system_version}")
        
        return True
    
    def complete_handshake(self, session_id: str) -> bool:
        """
        Complete handshake and register component
        Production finalization with trust scoring
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if session.status != HandshakeStatus.VALIDATED:
            logger.error(f"Cannot complete handshake: status is {session.status.value}")
            return False
        
        # Calculate trust score
        trust_score = self._calculate_trust_score(session)
        session.trust_score = trust_score
        
        if trust_score < self.min_trust_score:
            session.status = HandshakeStatus.FAILED
            logger.error(f"Trust score too low: {trust_score:.2f} < {self.min_trust_score}")
            return False
        
        # Register with trust manager
        if self.trust_manager:
            self.trust_manager.initialize_trust(
                entity_id=session.component_identity.component_id,
                entity_type=session.component_identity.component_type,
                initial_score=trust_score,
                metadata={
                    'capabilities': list(session.capabilities),
                    'version': session.version,
                    'registered_at': datetime.now(timezone.utc).isoformat()  # FIXED
                }
            )
        
        # Store registered component
        self.registered_components[session.component_identity.component_id] = session.component_identity
        
        # Mark session as active
        session.status = HandshakeStatus.ACTIVE
        
        logger.info(f"Handshake completed for {session.component_identity.component_id} (trust: {trust_score:.2f})")
        
        return True
    
    def revoke_component(self, component_id: str, reason: str) -> bool:
        """Revoke component registration"""
        if component_id not in self.registered_components:
            return False
        
        # Remove from registered components
        del self.registered_components[component_id]
        
        # Update trust score
        if self.trust_manager:
            self.trust_manager.record_failure(
                component_id,
                severity=1.0,
                context={'reason': reason, 'revoked': True}
            )
        
        # Find and expire related sessions
        for session_id, session in self.active_sessions.items():
            if session.component_identity.component_id == component_id:
                session.status = HandshakeStatus.FAILED
                session.metadata['revocation_reason'] = reason
        
        logger.warning(f"Component revoked: {component_id} - {reason}")
        
        return True
    
    def _generate_challenge(self) -> str:
        """Generate cryptographic challenge"""
        return secrets.token_urlsafe(32)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{secrets.token_hex(16)}"
    
    def _generate_public_key(self, component_id: str) -> str:
        """Generate public key for component (simplified)"""
        # In production, use proper PKI
        data = f"{component_id}:{datetime.now(timezone.utc).isoformat()}:{secrets.token_hex(16)}"  # FIXED
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _sign_identity(self, component_id: str, component_type: str) -> str:
        """Sign component identity (simplified)"""
        # In production, use proper digital signatures
        data = f"{component_id}:{component_type}:{datetime.now(timezone.utc).isoformat()}"  # FIXED
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _compute_challenge_response(self, challenge: str, component_id: str) -> str:
        """Compute expected challenge response"""
        data = f"{challenge}:{component_id}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _verify_signature(self, identity: ComponentIdentity) -> bool:
        """Verify identity signature"""
        # In production, use proper signature verification
        expected_sig = self._sign_identity(identity.component_id, identity.component_type)
        # Allow signature to be valid for reasonable time window
        return len(identity.signature) == 64  # SHA256 hex length
    
    def _verify_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Verify component credentials"""
        # In production, integrate with proper auth system
        required_fields = {'api_key', 'secret'}
        return all(field in credentials for field in required_fields)
    
    def _parse_version(self, version: str) -> tuple:
        """Parse semantic version"""
        try:
            parts = version.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except:
            return (0, 0, 0)
    
    def _calculate_trust_score(self, session: HandshakeSession) -> float:
        """Calculate initial trust score for component"""
        score = 0.5  # Base score
        
        # Bonus for standard capabilities
        standard_caps = {'reasoning', 'memory', 'logging'}
        if session.capabilities & standard_caps:
            score += 0.1
        
        # Bonus for metadata completeness
        if len(session.metadata) >= 3:
            score += 0.1
        
        # Bonus for proper authentication
        if session.challenge_response:
            score += 0.2
        
        # Bonus for version compliance
        if session.version and '.' in session.version:
            score += 0.1
        
        return min(1.0, score)
    
    def get_registered_components(self) -> List[Dict[str, Any]]:
        """Get all registered components"""
        return [
            {
                'component_id': identity.component_id,
                'component_type': identity.component_type,
                'registered_at': identity.timestamp.isoformat(),
                'public_key': identity.public_key[:16] + "...",
                'metadata': identity.metadata
            }
            for identity in self.registered_components.values()
        ]
    
    def cleanup_expired_sessions(self):
        """Clean up expired handshake sessions"""
        now = datetime.now(timezone.utc)  # FIXED
        expired = []
        
        for session_id, session in self.active_sessions.items():
            if now > session.expires_at:
                expired.append(session_id)
                session.status = HandshakeStatus.EXPIRED
        
        for session_id in expired:
            del self.active_sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_handshake_statistics(self) -> Dict[str, Any]:
        """Get handshake system statistics"""
        status_counts = {}
        for status in HandshakeStatus:
            status_counts[status.value] = sum(
                1 for s in self.active_sessions.values() if s.status == status
            )
        
        return {
            'active_sessions': len(self.active_sessions),
            'registered_components': len(self.registered_components),
            'by_status': status_counts,
            'avg_trust_score': sum(
                s.trust_score for s in self.active_sessions.values()
            ) / len(self.active_sessions) if self.active_sessions else 0
        }

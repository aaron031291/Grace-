"""
Security tests - RBAC, encryption, rate limiting
"""

import pytest
import asyncio

from grace.security import RBACManager, EncryptionManager, RateLimiter, Role, Permission
from grace.security.rate_limiter import RateLimitExceeded


@pytest.mark.asyncio
async def test_rbac_role_assignment():
    """Assert RBAC assigns roles correctly"""
    rbac = RBACManager()
    
    success = await rbac.assign_role(
        user_id="user123",
        role=Role.USER,
        assigned_by="admin",
        trust_score=0.9
    )
    
    assert success is True
    assert Role.USER in rbac.user_roles["user123"]


@pytest.mark.asyncio
async def test_rbac_permission_check():
    """Assert RBAC checks permissions correctly"""
    rbac = RBACManager()
    
    await rbac.assign_role("user123", Role.USER, "admin")
    
    # User should have read permissions
    assert rbac.has_permission("user123", Permission.READ_EVENTS) is True
    
    # User should NOT have admin permissions
    assert rbac.has_permission("user123", Permission.SYSTEM_ADMIN) is False


@pytest.mark.asyncio
async def test_rbac_constitutional_validation():
    """Assert privileged roles require constitutional validation"""
    from grace.governance.engine import GovernanceEngine
    
    governance = GovernanceEngine()
    rbac = RBACManager(governance_engine=governance)
    
    # Admin role requires constitutional validation
    success = await rbac.assign_role(
        user_id="privileged_user",
        role=Role.ADMIN,
        assigned_by="system",
        trust_score=0.95
    )
    
    # Should succeed with high trust score
    assert success is True


def test_encryption_basic():
    """Assert encryption works correctly"""
    enc = EncryptionManager()
    
    plaintext = "sensitive data"
    encrypted = enc.encrypt(plaintext)
    
    assert encrypted != plaintext
    assert len(encrypted) > 0
    
    decrypted = enc.decrypt(encrypted)
    assert decrypted == plaintext


def test_encryption_dict():
    """Assert dictionary field encryption works"""
    enc = EncryptionManager()
    
    data = {
        "username": "alice",
        "password": "secret123",
        "api_key": "key_abc"
    }
    
    encrypted = enc.encrypt_dict(data, ["password", "api_key"])
    
    assert encrypted["username"] == "alice"
    assert encrypted["password"] != "secret123"
    assert encrypted["api_key"] != "key_abc"
    assert encrypted["password_encrypted"] is True
    
    decrypted = enc.decrypt_dict(encrypted, ["password", "api_key"])
    
    assert decrypted["password"] == "secret123"
    assert decrypted["api_key"] == "key_abc"


def test_password_hashing():
    """Assert password hashing is secure"""
    password = "mySecurePassword123!"
    
    hashed, salt = EncryptionManager.hash_password(password)
    
    assert hashed != password
    assert len(salt) > 0
    
    # Verify correct password
    assert EncryptionManager.verify_password(password, hashed, salt) is True
    
    # Verify wrong password
    assert EncryptionManager.verify_password("wrongPassword", hashed, salt) is False


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_limit():
    """Assert rate limiter allows requests within limit"""
    limiter = RateLimiter(default_limit=10, default_window=60)
    
    # Should allow 10 requests
    for i in range(10):
        result = await limiter.check_rate_limit("user123", "test_endpoint")
        assert result is True


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    """Assert rate limiter blocks requests over limit"""
    limiter = RateLimiter(default_limit=5, default_window=60)
    
    # Use up all tokens
    for i in range(5):
        await limiter.check_rate_limit("user123", "test_endpoint")
    
    # Next request should be blocked
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check_rate_limit("user123", "test_endpoint")
    
    assert exc_info.value.limit == 5
    assert exc_info.value.window == 60


@pytest.mark.asyncio
async def test_rate_limiter_per_user():
    """Assert rate limiter is per-user"""
    limiter = RateLimiter(default_limit=5, default_window=60)
    
    # User1 uses all tokens
    for i in range(5):
        await limiter.check_rate_limit("user1", "test")
    
    # User2 should still have tokens
    result = await limiter.check_rate_limit("user2", "test")
    assert result is True


@pytest.mark.asyncio
async def test_ingress_kernel_security():
    """Integration test: Ingress kernel enforces security"""
    from grace.integration.event_bus import EventBus
    from grace.trigger_mesh import TriggerMesh
    from grace.governance.engine import GovernanceEngine
    from grace.ingress_kernel import IngressKernel
    
    bus = EventBus()
    mesh = TriggerMesh(bus)
    governance = GovernanceEngine()
    
    ingress = IngressKernel(
        event_bus=bus,
        trigger_mesh=mesh,
        governance_engine=governance,
        immutable_logs=None
    )
    
    # Assign user role
    await ingress.rbac.assign_role("test_user", Role.USER, "system")
    
    # Should allow permitted action
    result = await ingress.handle_request(
        user_id="test_user",
        endpoint="events/read",
        action="read",
        payload={},
        trust_score=0.9
    )
    
    assert result["success"] is True
    
    # Should deny unpermitted action
    result = await ingress.handle_request(
        user_id="test_user",
        endpoint="admin/users",
        action="manage",
        payload={},
        trust_score=0.9
    )
    
    assert result["success"] is False
    assert "Permission denied" in result["error"]

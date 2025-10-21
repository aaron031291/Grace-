"""
Test async memory layers
"""

import pytest
import asyncio
from datetime import datetime, timezone

from grace.memory.async_lightning import AsyncLightningMemory
from grace.memory.async_fusion import AsyncFusionMemory
from grace.memory.immutable_logs_async import AsyncImmutableLogs


@pytest.mark.asyncio
async def test_async_lightning_memory():
    """Test async Lightning memory (Redis)"""
    memory = AsyncLightningMemory()
    await memory.connect()
    
    # Test set/get
    await memory.set("test_key", {"data": "value"}, ttl=60)
    result = await memory.get("test_key")
    
    assert result == {"data": "value"}
    
    # Test exists
    exists = await memory.exists("test_key")
    assert exists is True
    
    # Test delete
    await memory.delete("test_key")
    result = await memory.get("test_key")
    assert result is None
    
    await memory.disconnect()


@pytest.mark.asyncio
async def test_async_fusion_memory():
    """Test async Fusion memory (Postgres)"""
    memory = AsyncFusionMemory("postgresql://localhost/grace_test")
    
    try:
        await memory.connect()
        
        # Test pattern storage
        pattern_id = await memory.store_pattern(
            pattern_type="behavior",
            pattern_data={"action": "click", "frequency": 10},
            confidence=0.85
        )
        
        assert pattern_id > 0
        
        # Test pattern retrieval
        patterns = await memory.get_patterns(pattern_type="behavior")
        assert len(patterns) > 0
        
        # Test interaction logging
        interaction_id = await memory.record_interaction(
            action="login",
            user_id="user123",
            context={"ip": "127.0.0.1"},
            outcome="success"
        )
        
        assert interaction_id > 0
        
        # Test audit logging
        log_id = await memory.log_audit_event(
            event_type="security",
            event_data={"action": "authentication"},
            actor="system",
            severity="info"
        )
        
        assert log_id > 0
        
        await memory.disconnect()
    
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")


@pytest.mark.asyncio
async def test_async_immutable_logs():
    """Test async immutable logs"""
    logs = AsyncImmutableLogs("postgresql://localhost/grace_test")
    
    try:
        await logs.connect()
        
        # Log entries
        hash1 = await logs.log(
            operation_type="user_action",
            actor="user123",
            action={"type": "login"},
            result={"success": True}
        )
        
        hash2 = await logs.log(
            operation_type="user_action",
            actor="user123",
            action={"type": "data_access"},
            result={"records": 10}
        )
        
        # Force flush
        await logs._flush_batch()
        
        # Verify chain
        verification = await logs.verify_chain()
        assert verification["verified"] is True
        
        # Query logs
        entries = await logs.query(actor="user123")
        assert len(entries) >= 2
        
        await logs.disconnect()
    
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")

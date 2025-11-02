#!/usr/bin/env python3
"""
Hunter Protocol - End-to-End Integration Test
=============================================

Tests the complete 17-stage pipeline with real data.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHunterPipeline:
    """Test complete Hunter Protocol pipeline"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_code_submission(self):
        """Test full pipeline with code submission"""
        from grace.hunter.pipeline import HunterPipeline
        
        pipeline = HunterPipeline()
        
        # Sample Python code with hunter marker
        code = b'''# (hunter)
def fibonacci(n):
    """Calculate Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci():
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
'''
        
        metadata = {
            "name": "fibonacci",
            "version": "1.0.0",
            "owner": "test_user",
            "type": "code"
        }
        
        # Process through pipeline
        context = await pipeline.process(code, metadata)
        
        # Verify stages completed
        assert len(context.completed_stages) == 17, f"Expected 17 stages, got {len(context.completed_stages)}"
        
        # Verify key results
        assert context.correlation_id is not None
        assert context.has_hunter_marker is True
        assert context.data_type.value == "code"
        assert context.security_passed is True
        assert context.quality_score > 0
        assert context.trust_score > 0
        assert context.final_validation_passed is True
        assert context.module_id is not None
        
        # Verify deployment
        assert len(context.endpoints) > 0
        assert context.deployed_at is not None
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Correlation ID: {context.correlation_id}")
        print(f"   Module ID: {context.module_id}")
        print(f"   Trust Score: {context.trust_score:.2f}")
        print(f"   Quality Score: {context.quality_score:.2f}")
        print(f"   Stages: {len(context.completed_stages)}/17")
    
    @pytest.mark.asyncio
    async def test_pipeline_rejects_dangerous_code(self):
        """Test that pipeline rejects dangerous code"""
        from grace.hunter.pipeline import HunterPipeline
        
        pipeline = HunterPipeline()
        
        # Code with dangerous pattern
        dangerous_code = b'''# (hunter)
import os
os.system("rm -rf /")  # Dangerous!
'''
        
        metadata = {"name": "dangerous", "version": "1.0", "owner": "bad_actor", "type": "code"}
        
        context = await pipeline.process(dangerous_code, metadata)
        
        # Should be rejected
        assert context.security_passed is False
        assert len(context.security_violations) > 0
        assert context.final_validation_passed is False
        
        print(f"\n✅ Dangerous code correctly rejected!")
        print(f"   Security violations: {len(context.security_violations)}")
    
    @pytest.mark.asyncio
    async def test_pipeline_requires_hunter_marker(self):
        """Test that pipeline requires hunter marker for code"""
        from grace.hunter.pipeline import HunterPipeline
        
        pipeline = HunterPipeline()
        
        # Code without hunter marker
        code_no_marker = b'''
def test():
    return "no marker"
'''
        
        metadata = {"name": "no_marker", "version": "1.0", "owner": "user", "type": "code"}
        
        context = await pipeline.process(code_no_marker, metadata)
        
        # Should fail at stage 2
        assert context.has_hunter_marker is False
        assert len(context.errors) > 0
        assert "marker" in context.errors[0].lower()
        
        print(f"\n✅ Missing hunter marker correctly detected!")


class TestHunterStages:
    """Test individual Hunter Protocol stages"""
    
    @pytest.mark.asyncio
    async def test_stage1_ingestion(self):
        """Test Stage 1: Ingestion"""
        from grace.hunter.stages import stage_1_ingestion
        
        result = await stage_1_ingestion.process(
            b"test data",
            {"name": "test"}
        )
        
        assert result.passed is True
        assert "correlation_id" in result.data
        assert "content_hash" in result.data
        assert result.data["size_bytes"] == 9
    
    @pytest.mark.asyncio
    async def test_stage2_hunter_marker(self):
        """Test Stage 2: Hunter Marker"""
        from grace.hunter.stages import stage_2_hunter_marker
        
        # With marker
        context = {
            "raw_data": b"# (hunter)\ndef test(): pass",
            "metadata": {"type": "code"}
        }
        result = await stage_2_hunter_marker.process(context)
        assert result.passed is True
        
        # Without marker
        context_no_marker = {
            "raw_data": b"def test(): pass",
            "metadata": {"type": "code"}
        }
        result = await stage_2_hunter_marker.process(context_no_marker)
        assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_stage6_security(self):
        """Test Stage 6: Security Validation"""
        from grace.hunter.stages import stage_6_security
        
        # Safe code
        safe_context = {"raw_data": b"def test(): pass"}
        result = await stage_6_security.process(safe_context)
        assert result.passed is True
        assert len(result.data["violations"]) == 0
        
        # Dangerous code
        dangerous_context = {"raw_data": b"import os\nos.system('ls')"}
        result = await stage_6_security.process(dangerous_context)
        assert result.passed is False
        assert len(result.data["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_stage10_trust_scoring(self):
        """Test Stage 10: Trust Scoring"""
        from grace.hunter.stages import stage_10_trust
        
        context = {
            "security_risk_score": 0.1,
            "quality_score": 0.85,
            "schema_validation": {"passed": True}
        }
        
        result = await stage_10_trust.process(context)
        
        assert result.passed is True
        assert "trust_score" in result.data
        assert 0.0 <= result.data["trust_score"] <= 1.0
        assert result.data["trust_level"] in ["untrusted", "low", "medium", "high"]


class TestHunterAdapters:
    """Test Hunter data adapters"""
    
    @pytest.mark.asyncio
    async def test_code_adapter(self):
        """Test code adapter"""
        from grace.hunter.adapters import CodeAdapter
        
        adapter = CodeAdapter()
        
        code = b'''
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    pass
'''
        
        result = await adapter.process(code, {})
        
        assert result["success"] is True
        assert result["language"] == "python"
        assert len(result["functions"]) > 0
        assert len(result["classes"]) > 0
    
    @pytest.mark.asyncio
    async def test_document_adapter(self):
        """Test document adapter"""
        from grace.hunter.adapters import DocumentAdapter
        
        adapter = DocumentAdapter()
        
        doc = b"This is a test document with some content."
        
        result = await adapter.process(doc, {})
        
        assert result["success"] is True
        assert result["word_count"] > 0
        assert len(result["chunks"]) > 0
    
    @pytest.mark.asyncio
    async def test_structured_adapter(self):
        """Test structured data adapter"""
        from grace.hunter.adapters import StructuredAdapter
        
        adapter = StructuredAdapter()
        
        json_data = b'{"name": "test", "value": 123}'
        
        result = await adapter.process(json_data, {})
        
        assert result["success"] is True
        assert result["format"] == "json"
        assert result["data"]["name"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

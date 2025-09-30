"""
Test file to verify governance enforcement works correctly.

This file should pass all governance checks as an example of compliant code.
"""
import asyncio
from typing import Dict, Any
from grace.governance.constitutional_decorator import constitutional_check, trust_middleware
from grace.audit.golden_path_auditor import append_audit


@constitutional_check(policy="default", transparency_level="public")
async def safe_operation(data: Dict[str, Any], rationale: str = "Safe data processing") -> Dict[str, Any]:
    """
    A safe operation that complies with governance requirements.
    
    This function demonstrates proper governance integration:
    - Uses constitutional decorator
    - Provides rationale parameter
    - Returns governance metadata
    """
    result = {
        "processed_data": data,
        "status": "success",
        "timestamp": "2025-01-01T00:00:00Z"
    }
    
    return result


@trust_middleware(min_trust_score=0.7)
@constitutional_check(policy="default")
async def trusted_operation(data: Dict[str, Any], rationale: str = "Trusted computation") -> Dict[str, Any]:
    """
    An operation requiring both trust and constitutional compliance.
    """
    # Log the operation for audit trail
    audit_id = await append_audit(
        operation_type="trusted_computation",
        operation_data={"operation": "data_processing", "data_size": len(str(data))},
        transparency_level="democratic_oversight"
    )
    
    return {
        "result": "computed",
        "audit_id": audit_id,
        "trust_verified": True
    }


class CompliantService:
    """
    Example of a service that follows governance best practices.
    """
    
    def __init__(self):
        self.service_name = "CompliantService"
        
    @constitutional_check(policy="default")
    async def public_method(self, data: Dict[str, Any], rationale: str = "Public service operation") -> Dict[str, Any]:
        """Public method with governance compliance."""
        return {"service": self.service_name, "data": data}
    
    async def safe_read_operation(self, query: str) -> Dict[str, Any]:
        """Safe read operation that doesn't need special governance."""
        # Regular read operations can be simpler
        audit_id = await append_audit(
            operation_type="safe_read",
            operation_data={"query": query},
            transparency_level="public"
        )
        
        return {
            "query": query,
            "results": ["example_result_1", "example_result_2"],
            "audit_id": audit_id
        }


async def main():
    """Demonstrate compliant governance usage."""
    
    # Test constitutional decorator
    result1 = await safe_operation(
        {"test": "data"}, 
        rationale="Testing governance compliance"
    )
    print(f"Safe operation result: {result1}")
    
    # Test combined trust and constitutional checks
    result2 = await trusted_operation(
        {"sensitive": "data"},
        rationale="Processing sensitive data with full governance"
    )
    print(f"Trusted operation result: {result2}")
    
    # Test service methods
    service = CompliantService()
    result3 = await service.public_method(
        {"service": "test"},
        rationale="Public service usage"
    )
    print(f"Service method result: {result3}")
    
    result4 = await service.safe_read_operation("test query")
    print(f"Safe read result: {result4}")


if __name__ == "__main__":
    asyncio.run(main())
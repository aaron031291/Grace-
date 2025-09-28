"""
Grace Communications Validator - Schema validation for GME messages.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from .envelope import GraceMessageEnvelope

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of message validation."""
    def __init__(self, passed: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.passed = passed
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


def validate_envelope(envelope: Dict[str, Any]) -> ValidationResult:
    """
    Validate a Grace Message Envelope against the GME schema.
    
    Args:
        envelope: Dictionary representation of the envelope
        
    Returns:
        ValidationResult with pass/fail status and any errors/warnings
    """
    result = ValidationResult()
    
    try:
        # Try to parse as GME using Pydantic
        gme = GraceMessageEnvelope.model_validate(envelope)
        
        # Additional validations
        _validate_payload_consistency(gme, result)
        _validate_headers(gme.headers, result)
        _validate_governance_requirements(gme, result)
        
    except Exception as e:
        result.add_error(f"GME validation failed: {str(e)}")
    
    return result


def validate_payload(payload: Dict[str, Any], schema_ref: str) -> ValidationResult:
    """
    Validate payload against its referenced schema.
    
    Args:
        payload: The payload to validate
        schema_ref: Schema reference URI
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    # For now, basic validation - in production would load actual schemas
    if not payload:
        result.add_error("Payload cannot be empty")
        return result
    
    # Basic type checking
    if not isinstance(payload, dict):
        result.add_error("Payload must be an object/dictionary")
        return result
    
    # Check for common required fields based on schema_ref
    if "intelligence" in schema_ref.lower():
        _validate_intelligence_payload(payload, result)
    elif "mldl" in schema_ref.lower():
        _validate_mldl_payload(payload, result)
    elif "governance" in schema_ref.lower():
        _validate_governance_payload(payload, result)
    
    return result


def _validate_payload_consistency(gme: GraceMessageEnvelope, result: ValidationResult):
    """Validate payload/payload_ref consistency."""
    has_payload = gme.payload is not None
    has_payload_ref = gme.payload_ref is not None
    
    if not has_payload and not has_payload_ref:
        result.add_error("Either payload or payload_ref must be provided")
    elif has_payload and has_payload_ref:
        result.add_error("Only one of payload or payload_ref should be provided")
    
    # Check payload size recommendations
    if has_payload and gme.payload:
        payload_size = len(json.dumps(gme.payload).encode('utf-8'))
        if payload_size > 262144:  # 256KB
            result.add_warning(f"Large payload ({payload_size} bytes) should use payload_ref")


def _validate_headers(headers, result: ValidationResult):
    """Validate message headers."""
    # Check correlation ID format
    if not headers.correlation_id.startswith("cor_"):
        result.add_error("correlation_id must start with 'cor_'")
    
    # Check partition key is set
    if not headers.partition_key:
        result.add_error("partition_key is required")
    
    # Validate priority/QoS combinations
    if headers.priority == "P0" and headers.qos != "realtime":
        result.add_warning("P0 priority should typically use realtime QoS")
    
    # Check hop count
    if headers.hop_count > 32:
        result.add_warning(f"High hop count ({headers.hop_count}) may indicate routing loops")


def _validate_governance_requirements(gme: GraceMessageEnvelope, result: ValidationResult):
    """Validate governance and security requirements."""
    
    # Check if governance approval is required
    governance_required_events = [
        "MLDL_DEPLOYMENT_REQUESTED",
        "POLICY_CHANGE",
        "SECURITY_EXCEPTION"
    ]
    
    if gme.name in governance_required_events:
        if "gov.approve" not in gme.headers.rbac:
            result.add_error(f"Event {gme.name} requires governance approval RBAC")
    
    # Check PII handling
    if gme.headers.pii_flags and not gme.headers.consent_scope:
        result.add_error("Messages with PII flags must specify consent scope")
    
    # Check restricted data handling
    if gme.headers.governance_label == "restricted":
        if not gme.headers.signature:
            result.add_warning("Restricted messages should be signed")
        if not gme.headers.checksum:
            result.add_warning("Restricted messages should include checksum")


def _validate_intelligence_payload(payload: Dict[str, Any], result: ValidationResult):
    """Validate intelligence-specific payload requirements."""
    if "query" in payload:
        if not payload.get("context"):
            result.add_warning("Intelligence queries should include context")
    
    if "result" in payload:
        if "confidence" not in payload.get("result", {}):
            result.add_warning("Intelligence results should include confidence scores")


def _validate_mldl_payload(payload: Dict[str, Any], result: ValidationResult):
    """Validate MLDL-specific payload requirements."""
    if "model_key" in payload and not payload.get("version"):
        result.add_error("MLDL payloads with model_key must specify version")
    
    if "deployment" in payload:
        required_fields = ["environment", "model_key", "version"]
        for field in required_fields:
            if field not in payload.get("deployment", {}):
                result.add_error(f"MLDL deployment payload missing required field: {field}")


def _validate_governance_payload(payload: Dict[str, Any], result: ValidationResult):
    """Validate governance-specific payload requirements.""" 
    if "decision" in payload:
        decision = payload["decision"]
        if "approved" not in decision and "rejected" not in decision:
            result.add_error("Governance decisions must specify approved or rejected")
        
        if not decision.get("reasoning"):
            result.add_warning("Governance decisions should include reasoning")


def get_schema_path(schema_ref: str) -> Optional[Path]:
    """Get local file path for a schema reference."""
    # Convert grace:// URIs to local paths
    if schema_ref.startswith("grace://contracts/"):
        rel_path = schema_ref.replace("grace://contracts/", "")
        return Path(__file__).parent.parent.parent / "contracts" / rel_path
    return None
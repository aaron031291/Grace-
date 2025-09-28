"""
Policy Validators - Enforce contracts, PII, format, and governance policies.
"""
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from grace.contracts.ingress_contracts import NormRecord, SourceConfig


logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of policy validation."""
    
    def __init__(self, passed: bool, policy_type: str, 
                 violations: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None,
                 actions_taken: Optional[List[str]] = None):
        self.passed = passed
        self.policy_type = policy_type
        self.violations = violations or []
        self.warnings = warnings or []
        self.actions_taken = actions_taken or []


class BaseValidator(ABC):
    """Base class for policy validators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def validate(self, record: NormRecord, source_config: SourceConfig) -> ValidationResult:
        """Validate record against policy."""
        pass


class PIIValidator(BaseValidator):
    """Validates and enforces PII policies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # PII detection patterns
        self.pii_patterns = {
            "ssn": [
                r'\b\d{3}-?\d{2}-?\d{4}\b',
                r'\b\d{3}\s\d{2}\s\d{4}\b'
            ],
            "credit_card": [
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                r'\b\d{13,19}\b'
            ],
            "email": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "phone": [
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            ],
            "ip_address": [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ],
            "api_key": [
                r'\b[A-Za-z0-9]{32,}\b',
                r'(?i)(?:api.?key|token|secret)[\s:=]+[A-Za-z0-9_\-]{10,}'
            ]
        }
    
    async def validate(self, record: NormRecord, source_config: SourceConfig) -> ValidationResult:
        """Validate PII policy compliance."""
        try:
            pii_flags = await self._detect_pii(record)
            pii_policy = source_config.pii_policy
            
            violations = []
            warnings = []
            actions_taken = []
            
            if pii_flags:
                if pii_policy == "block":
                    violations.append(f"PII detected with block policy: {', '.join(pii_flags)}")
                    return ValidationResult(
                        passed=False,
                        policy_type="pii",
                        violations=violations
                    )
                
                elif pii_policy == "mask":
                    # Apply masking
                    await self._mask_pii(record, pii_flags)
                    actions_taken.append(f"Masked PII: {', '.join(pii_flags)}")
                
                elif pii_policy == "hash":
                    # Apply hashing
                    await self._hash_pii(record, pii_flags)
                    actions_taken.append(f"Hashed PII: {', '.join(pii_flags)}")
                
                elif pii_policy == "allow_with_consent":
                    # Check for consent indicators
                    if not await self._check_consent(record):
                        violations.append("PII detected without proper consent")
                        return ValidationResult(
                            passed=False,
                            policy_type="pii",
                            violations=violations
                        )
                    warnings.append(f"PII allowed with consent: {', '.join(pii_flags)}")
            
            # Update record's PII flags
            record.quality.pii_flags = pii_flags
            
            return ValidationResult(
                passed=True,
                policy_type="pii",
                warnings=warnings,
                actions_taken=actions_taken
            )
            
        except Exception as e:
            logger.error(f"PII validation failed: {e}")
            return ValidationResult(
                passed=False,
                policy_type="pii",
                violations=[f"PII validation error: {str(e)}"]
            )
    
    async def _detect_pii(self, record: NormRecord) -> List[str]:
        """Detect PII in record content."""
        detected_pii = []
        
        # Convert record body to searchable text
        text_content = self._extract_text_from_record(record)
        
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    detected_pii.append(pii_type)
                    break  # Don't add duplicate types
        
        return detected_pii
    
    def _extract_text_from_record(self, record: NormRecord) -> str:
        """Extract searchable text from record body."""
        text_parts = []
        
        def extract_strings(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_strings(value, depth + 1)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_strings(item, depth + 1)
        
        extract_strings(record.body)
        return " ".join(text_parts)
    
    async def _mask_pii(self, record: NormRecord, pii_types: List[str]):
        """Apply masking to PII in record."""
        text_content = self._extract_text_from_record(record)
        
        for pii_type in pii_types:
            patterns = self.pii_patterns.get(pii_type, [])
            for pattern in patterns:
                # Replace with masked version
                if pii_type == "ssn":
                    text_content = re.sub(pattern, "***-**-****", text_content)
                elif pii_type == "credit_card":
                    text_content = re.sub(pattern, "**** **** **** ****", text_content)
                elif pii_type == "email":
                    text_content = re.sub(pattern, "***@***.***", text_content)
                elif pii_type == "phone":
                    text_content = re.sub(pattern, "***-***-****", text_content)
                else:
                    text_content = re.sub(pattern, "*" * 8, text_content)
        
        # Update record body with masked content
        self._update_record_with_masked_text(record, text_content)
    
    async def _hash_pii(self, record: NormRecord, pii_types: List[str]):
        """Apply hashing to PII in record."""
        import hashlib
        text_content = self._extract_text_from_record(record)
        
        for pii_type in pii_types:
            patterns = self.pii_patterns.get(pii_type, [])
            for pattern in patterns:
                def hash_match(match):
                    original = match.group(0)
                    hashed = hashlib.sha256(original.encode()).hexdigest()[:16]
                    return f"[HASH:{hashed}]"
                
                text_content = re.sub(pattern, hash_match, text_content)
        
        # Update record body with hashed content
        self._update_record_with_masked_text(record, text_content)
    
    async def _check_consent(self, record: NormRecord) -> bool:
        """Check for consent indicators in record."""
        text_content = self._extract_text_from_record(record).lower()
        
        consent_indicators = [
            "consent", "permission", "authorized", "opt-in", 
            "agreed", "accepted terms", "privacy policy"
        ]
        
        return any(indicator in text_content for indicator in consent_indicators)
    
    def _update_record_with_masked_text(self, record: NormRecord, masked_text: str):
        """Update record body with masked/hashed text."""
        # Simple implementation - in practice would need more sophisticated text replacement
        if "text" in record.body:
            record.body["text"] = masked_text[:len(record.body["text"])]
        elif "content" in record.body:
            record.body["content"] = masked_text[:len(record.body["content"])]


class SchemaValidator(BaseValidator):
    """Validates records against contract schemas."""
    
    async def validate(self, record: NormRecord, source_config: SourceConfig) -> ValidationResult:
        """Validate record against target contract schema."""
        try:
            contract = source_config.target_contract
            violations = []
            warnings = []
            
            # Get schema for contract
            schema = await self._get_contract_schema(contract)
            if not schema:
                violations.append(f"Unknown contract schema: {contract}")
                return ValidationResult(
                    passed=False,
                    policy_type="schema",
                    violations=violations
                )
            
            # Validate required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in record.body:
                    violations.append(f"Missing required field: {field}")
            
            # Validate field types and formats
            properties = schema.get("properties", {})
            for field, value in record.body.items():
                if field in properties:
                    field_schema = properties[field]
                    field_violations = await self._validate_field(field, value, field_schema)
                    violations.extend(field_violations)
            
            # Calculate validity score
            total_checks = len(required_fields) + len(properties)
            if total_checks > 0:
                validity_score = max(0.0, 1.0 - (len(violations) / total_checks))
                record.quality.validity_score = validity_score
            
            return ValidationResult(
                passed=len(violations) == 0,
                policy_type="schema",
                violations=violations,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return ValidationResult(
                passed=False,
                policy_type="schema",
                violations=[f"Schema validation error: {str(e)}"]
            )
    
    async def _get_contract_schema(self, contract: str) -> Optional[Dict[str, Any]]:
        """Get schema definition for contract."""
        # Mock schema definitions - in real implementation would load from registry
        schemas = {
            "contract:article.v1": {
                "required": ["title", "url", "text"],
                "properties": {
                    "title": {"type": "string", "maxLength": 500},
                    "author": {"type": ["string", "null"]},
                    "url": {"type": "string", "format": "uri"},
                    "text": {"type": "string", "minLength": 10},
                    "language": {"type": "string", "pattern": "[a-z]{2}"},
                    "published_at": {"type": ["string", "null"], "format": "date-time"}
                }
            },
            "contract:transcript.v1": {
                "required": ["media_id", "lang", "segments"],
                "properties": {
                    "media_id": {"type": "string"},
                    "lang": {"type": "string", "pattern": "[a-z]{2}"},
                    "segments": {"type": "array", "minItems": 1},
                    "duration_s": {"type": ["number", "null"], "minimum": 0}
                }
            }
        }
        
        return schemas.get(contract)
    
    async def _validate_field(self, field_name: str, value: Any, 
                            field_schema: Dict[str, Any]) -> List[str]:
        """Validate a single field against its schema."""
        violations = []
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            if not self._check_type(value, expected_type):
                violations.append(f"Field '{field_name}' has wrong type")
        
        # String validations
        if isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            pattern = field_schema.get("pattern")
            
            if min_length and len(value) < min_length:
                violations.append(f"Field '{field_name}' too short")
            if max_length and len(value) > max_length:
                violations.append(f"Field '{field_name}' too long")
            if pattern and not re.match(pattern, value):
                violations.append(f"Field '{field_name}' doesn't match pattern")
        
        # Numeric validations
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            
            if minimum is not None and value < minimum:
                violations.append(f"Field '{field_name}' below minimum")
            if maximum is not None and value > maximum:
                violations.append(f"Field '{field_name}' above maximum")
        
        # Array validations
        if isinstance(value, list):
            min_items = field_schema.get("minItems")
            max_items = field_schema.get("maxItems")
            
            if min_items and len(value) < min_items:
                violations.append(f"Field '{field_name}' has too few items")
            if max_items and len(value) > max_items:
                violations.append(f"Field '{field_name}' has too many items")
        
        return violations
    
    def _check_type(self, value: Any, expected_type: Any) -> bool:
        """Check if value matches expected type."""
        if isinstance(expected_type, list):
            return any(self._check_type(value, t) for t in expected_type)
        
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "null":
            return value is None
        
        return True  # Unknown type, allow


class GovernanceValidator(BaseValidator):
    """Validates governance compliance."""
    
    async def validate(self, record: NormRecord, source_config: SourceConfig) -> ValidationResult:
        """Validate governance policy compliance."""
        try:
            violations = []
            warnings = []
            actions_taken = []
            
            # Check retention policy
            retention_days = source_config.retention_days
            max_allowed = self.config.get("max_retention_days", 2555)  # ~7 years
            
            if retention_days > max_allowed:
                violations.append(f"Retention period {retention_days} exceeds maximum {max_allowed}")
            
            # Check governance label compliance
            label = source_config.governance_label
            restricted_patterns = self.config.get("restricted_patterns", [
                "classified", "confidential", "secret", "restricted"
            ])
            
            content_text = self._extract_text_content(record).lower()
            
            if label == "public":
                # Public data shouldn't contain restricted content
                for pattern in restricted_patterns:
                    if pattern in content_text:
                        violations.append(f"Public data contains restricted content: {pattern}")
            
            elif label == "restricted":
                # Restricted data requires special handling
                actions_taken.append("Applied restricted data handling")
                warnings.append("Processing restricted data - audit trail required")
            
            # Check for export control violations
            export_control_patterns = self.config.get("export_control_patterns", [
                "itar", "ear", "dual use", "export control"
            ])
            
            for pattern in export_control_patterns:
                if pattern in content_text:
                    warnings.append(f"Potential export control content: {pattern}")
            
            return ValidationResult(
                passed=len(violations) == 0,
                policy_type="governance",
                violations=violations,
                warnings=warnings,
                actions_taken=actions_taken
            )
            
        except Exception as e:
            logger.error(f"Governance validation failed: {e}")
            return ValidationResult(
                passed=False,
                policy_type="governance",
                violations=[f"Governance validation error: {str(e)}"]
            )
    
    def _extract_text_content(self, record: NormRecord) -> str:
        """Extract text content from record for pattern matching."""
        text_parts = []
        
        # Look for common text fields
        for field in ["text", "content", "body", "description", "title", "summary"]:
            if field in record.body and isinstance(record.body[field], str):
                text_parts.append(record.body[field])
        
        return " ".join(text_parts)


class PolicyGuard:
    """Main policy enforcement coordinator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize validators
        self.pii_validator = PIIValidator(config.get("pii", {}))
        self.schema_validator = SchemaValidator(config.get("schema", {}))
        self.governance_validator = GovernanceValidator(config.get("governance", {}))
    
    async def enforce_policies(self, record: NormRecord, source_config: SourceConfig) -> Tuple[bool, List[str]]:
        """
        Enforce all policies on a record.
        
        Args:
            record: Normalized record to validate
            source_config: Source configuration
            
        Returns:
            (passed, violations) tuple
        """
        try:
            all_violations = []
            all_warnings = []
            all_actions = []
            
            # Run all validators
            validators = [
                ("pii", self.pii_validator),
                ("schema", self.schema_validator),
                ("governance", self.governance_validator)
            ]
            
            for validator_name, validator in validators:
                try:
                    result = await validator.validate(record, source_config)
                    
                    if not result.passed:
                        all_violations.extend(result.violations)
                    
                    all_warnings.extend(result.warnings)
                    all_actions.extend(result.actions_taken)
                    
                    logger.debug(f"{validator_name} validation: {'PASS' if result.passed else 'FAIL'}")
                    
                except Exception as e:
                    logger.error(f"{validator_name} validator failed: {e}")
                    all_violations.append(f"{validator_name} validation error: {str(e)}")
            
            # Log results
            if all_violations:
                logger.warning(f"Policy violations for {record.record_id}: {all_violations}")
            
            if all_warnings:
                logger.info(f"Policy warnings for {record.record_id}: {all_warnings}")
            
            if all_actions:
                logger.info(f"Policy actions for {record.record_id}: {all_actions}")
            
            return len(all_violations) == 0, all_violations
            
        except Exception as e:
            logger.error(f"Policy enforcement failed: {e}")
            return False, [f"Policy enforcement error: {str(e)}"]
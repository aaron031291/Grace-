# Grace Policy Validation Report

❌ **FAILED** - Policy violations detected

## Summary
- Total operations analyzed: 2
- Policy violations: 4
- Constitutional violations: 4
- Governance warnings: 1
- Blocked operations: 1
- Highest severity: critical

## Policy Violations
### dangerous_code_execution (critical)
**Description:** Block dangerous code execution patterns
**Actions:** block, log, alert

### ide_apply_changes (medium)
**Description:** Gate IDE apply changes to sandbox branch with approval
**Actions:** require_sandbox_branch, require_policy_pass, require_human_approval

### ide_apply_changes (medium)
**Description:** Gate IDE apply changes to sandbox branch with approval
**Actions:** require_sandbox_branch, require_policy_pass, require_human_approval

### network_access (medium)
**Description:** Control network access operations
**Actions:** require_approval, log


## 🏛️ Constitutional Violations
### transparency (minor)
**Description:** Transparency violations: No rationale or description provided; No audit trail available
**Recommendation:** Provide clear rationale and ensure audit trail

### accountability (major)
**Description:** Accountability violations: No clear decision maker identified; Action is not traceable
**Recommendation:** Ensure clear accountability chain and traceability

### transparency (minor)
**Description:** Transparency violations: No rationale or description provided; No audit trail available
**Recommendation:** Provide clear rationale and ensure audit trail

### accountability (major)
**Description:** Accountability violations: No clear decision maker identified; Action is not traceable
**Recommendation:** Ensure clear accountability chain and traceability


## ⚠️ Governance Warnings
- **grace/api/api_service.py**: API endpoint may lack governance enforcement


## Blocked Operations
- **code_modification**: grace/governance/constitutional_decorator.py

---
Policy file: `/home/runner/work/Grace-/Grace-/grace/policy/default_policies.yml`

This report includes constitutional compliance checks and governance enforcement validation.
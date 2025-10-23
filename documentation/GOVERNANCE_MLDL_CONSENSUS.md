# Governance ↔ MLDL Consensus Pattern

## Overview

The Governance kernel can request consensus from the MLDL (Machine Learning / Deep Learning) kernel for complex decisions. This implements a request/response loop using TriggerMesh's `emit` and `wait_for` capabilities.

## Flow Diagram

```
┌─────────────────┐                    ┌─────────────────┐
│   Governance    │                    │   MLDL Kernel   │
│     Engine      │                    │   (Consensus)   │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │ 1. validate(event)                  │
         │    ↓ has violations                 │
         │                                      │
         │ 2. emit("mldl.consensus.request")   │
         │────────────────────────────────────>│
         │    payload: {                        │
         │      decision_context,               │
         │      options                         │
         │    }                                 │
         │                                      │
         │                              3. Process request
         │                                 ↓ Consult specialists:
         │                                   - Heuristic
         │                                   - LLM
         │                                   - Statistical
         │                                 ↓ Compute quorum
         │                                      │
         │ 4. wait_for response (timeout: 5s)  │
         │<────────────────────────────────────│
         │    emit("mldl.consensus.response")  │
         │    payload: {                        │
         │      consensus: {                    │
         │        recommendation,               │
         │        confidence,                   │
         │        specialists                   │
         │      }                                │
         │    }                                 │
         │                                      │
         │ 5. Apply consensus                   │
         │    ↓ Override violations if          │
         │      approve + high confidence       │
         │                                      │
         ▼                                      ▼
```

## Usage

### From Governance

```python
from grace.governance.engine import GovernanceEngine

governance = GovernanceEngine(trigger_mesh=trigger_mesh)

# Validate with MLDL consensus
result = await governance.validate(
    event,
    context={"severity": "high"},
    request_mldl_consensus=True  # Enable consensus
)

if result.decision:
    consensus = result.decision.get("mldl_consensus")
    if consensus:
        print(f"Recommendation: {consensus['recommendation']}")
        print(f"Confidence: {consensus['confidence']}")
```

### Consensus Request Format

```yaml
event_type: "mldl.consensus.request"
payload:
  decision_context:
    event_id: "evt_123"
    event_type: "system.error"
    source: "api_gateway"
    trust_score: 0.6
    priority: "high"
    violations: ["Trust score too low"]
    context: {"error_severity": "high"}
  options:
    - "approve"
    - "review"
    - "reject"
```

### Consensus Response Format

```yaml
event_type: "mldl.consensus.response"
payload:
  consensus:
    recommendation: "review"
    confidence: 0.75
    specialists:
      - name: "heuristic"
        vote: "review"
        confidence: 0.70
      - name: "llm"
        vote: "review"
        confidence: 0.75
      - name: "statistical"
        vote: "approve"
        confidence: 0.80
    quorum_method: "weighted_majority"
    timestamp: "2024-01-15T10:30:00Z"
  processing_time_ms: 150
```

## MLDL Specialists

The MLDL kernel uses multiple specialists for consensus:

### 1. Heuristic Specialist
- **Method**: Rule-based decision tree
- **Input**: Trust score, violations
- **Logic**:
  - Trust ≥ 0.8 → approve (confidence: 0.85)
  - Trust ≥ 0.5 → review (confidence: 0.70)
  - Trust < 0.5 → reject (confidence: 0.90)

### 2. LLM Specialist (Optional)
- **Method**: Large Language Model inference
- **Input**: Full decision context as prompt
- **Fallback**: Returns "review" with 0.50 confidence if unavailable

### 3. Statistical Specialist
- **Method**: Bayesian-style classification
- **Input**: Violation count, patterns
- **Logic**:
  - 0 violations → approve (confidence: 0.80)
  - 1-2 violations → review (confidence: 0.75)
  - 3+ violations → reject (confidence: 0.85)

## Quorum Algorithm

The consensus uses **weighted majority voting**:

```python
def quorum_vote(specialists):
    vote_weights = {}
    
    for specialist in specialists:
        vote = specialist["vote"]
        confidence = specialist["confidence"]
        vote_weights[vote] += confidence
    
    best_vote = max(vote_weights, key=vote_weights.get)
    consensus_confidence = vote_weights[best_vote] / total_weight
    
    return {
        "recommendation": best_vote,
        "confidence": consensus_confidence
    }
```

## Override Logic

Governance applies consensus based on confidence:

- **Approve + confidence > 0.7**: Clear all violations
- **Reject + confidence > 0.8**: Add consensus violation
- **Review**: Keep existing violations

## Timeout Handling

- **Default timeout**: 5 seconds
- **On timeout**: Consensus is `None`, validation continues without ML input
- **Graceful degradation**: System remains functional without consensus

## Configuration

In `config/trigger_mesh.yaml`:

```yaml
routes:
  - name: "mldl_consensus"
    pattern: "mldl.consensus.request"
    targets:
      - "mldl_kernel"
    timeout_seconds: 5
    priority: "high"
```

## Testing

```bash
# Run consensus tests
pytest tests/test_governance_mldl_consensus.py -v

# Test specific scenarios
pytest tests/test_governance_mldl_consensus.py::test_governance_requests_mldl_consensus -v
```

## Metrics

MLDL kernel tracks:
- `consensus_count`: Total consensus requests processed
- `inference_count`: Total inferences (includes consensus)
- Processing time per consensus request

```python
health = mldl_kernel.get_health()
print(f"Consensus requests: {health['consensus_count']}")
```

# Grace Intelligent Test Quality System

## 🎯 Overview

Grace now features an **adaptive, KPI-integrated test quality monitoring system** that goes beyond simple pass/fail counting. The system:

1. **Tracks Component Quality Scores** (0-100%) instead of raw counts
2. **Integrates with KPITrustMonitor** for historical trust tracking
3. **Triggers Self-Healing** via TriggerMesh/EventBus
4. **Uses 90% Threshold** for "passing" components
5. **Provides Clear Progress Metrics** toward system-wide quality goals

## 📊 Quality Scoring Model

### Quality Levels

| Level | Score Range | Status | Action |
|-------|-------------|--------|--------|
| 🌟 EXCELLENT | ≥95% | Peak performance | Maintain |
| ✅ PASSING | ≥90% | Meets threshold | Counts toward system success |
| ⚡ ACCEPTABLE | 70-90% | Functional | Needs improvement |
| ⚠️ DEGRADED | 50-70% | Poor | Triggers adaptive learning |
| 🔴 CRITICAL | <50% | Failing | Escalates to AVN healing |

### Score Calculation

```python
# 1. Calculate raw score
raw_score = pass_rate - (error_severity_penalty * 0.3)

# 2. Blend with historical trust
trust_adjusted_score = 0.8 * raw_score + 0.2 * kpi_trust_score

# 3. Determine status
if score >= 0.95: status = EXCELLENT
elif score >= 0.90: status = PASSING
elif score >= 0.70: status = ACCEPTABLE
elif score >= 0.50: status = DEGRADED
else: status = CRITICAL
```

### Error Severity Weights

- **LOW** (0.1): Warnings, deprecated usage
- **MEDIUM** (0.5): Assertion failures, expected errors
- **HIGH** (1.0): Unexpected exceptions, integration failures
- **CRITICAL** (2.0): System crashes, data corruption

## 🔄 Self-Healing Integration

The system automatically triggers remediation based on component status:

### CRITICAL (<50%)
**Action:** Escalate to AVN Memory Orchestrator
```python
Event: "test_quality.healing_required"
Payload: {
    "severity": "CRITICAL",
    "recommended_actions": [
        "Review recent code changes",
        "Check system resources",
        "Verify dependencies",
        "Run diagnostic tests"
    ],
    "escalate_to": "avn_core"
}
```

### DEGRADED (50-70%)
**Action:** Trigger Learning Kernel adaptive loops
```python
Event: "test_quality.adaptive_learning_required"
Payload: {
    "severity": "WARNING",
    "focus_areas": [
        "Analyze error patterns",
        "Review test coverage",
        "Improve edge case handling"
    ],
    "trigger": "learning_kernel"
}
```

### ACCEPTABLE (70-90%)
**Action:** Suggest specific improvements
```python
Event: "test_quality.improvement_suggested"
Payload: {
    "current_score": 0.82,
    "target_score": 0.90,
    "gap": 0.08,
    "suggestions": [
        "Improve quality by 8% to reach threshold",
        "Address high-severity errors first",
        "Review failing test patterns"
    ]
}
```

## 📈 System-Wide Metrics

### Traditional vs Quality-Based

**Traditional Metrics (Raw Counts):**
- Total Tests: 193
- Passed: 159 (82.4%)
- Failed: 0 (0.0%)
- Skipped: 34 (17.6%)

**Quality-Based Metrics (90% Threshold):**
- Total Components: 5
- Passing Components: 2 (40%)
- System Pass Rate: 40%
- Overall Quality: 82.7%

### Why Quality-Based is Better

1. **No Confusing Low Percentages**
   - ❌ Old: "12% passing, 25% passing..." - what does this mean?
   - ✅ New: "Component at 82% quality (8% gap to passing)"

2. **Clear Milestone Tracking**
   - Each component either passes (≥90%) or doesn't
   - System progresses to 100% as components cross threshold
   - Easy to understand: "2 of 5 components passing"

3. **Integrated Learning**
   - Components improve incrementally (tracked in history)
   - But only count toward system metrics when ≥90%
   - Encourages quality over quantity

4. **Adaptive Intelligence**
   - KPI trust scores influence quality metrics
   - Historical performance matters
   - Self-healing triggers automatically

## 🚀 Usage

### Run Tests with Quality Monitoring

```bash
# Standard run
export PYTHONPATH="/workspaces/Grace-:$PYTHONPATH"
pytest --tb=no -q

# Enable self-healing (default)
pytest --enable-self-healing

# Disable self-healing
pytest --no-self-healing
```

### View Quality Reports

```bash
# View latest report
python scripts/view_quality_report.py

# Show all components
python scripts/view_quality_report.py --all

# Summary only
python scripts/view_quality_report.py --summary-only

# Components only
python scripts/view_quality_report.py --components-only
```

### Access Detailed JSON Reports

```bash
# View latest report
cat test_reports/quality_report_*.json | jq '.summary'

# View components needing attention
cat test_reports/quality_report_*.json | jq '.needing_attention'

# View component details
cat test_reports/quality_report_*.json | jq '.components.mcp_framework'
```

## 📊 Current Status (Latest Report)

### System Summary
- **System Pass Rate:** 40% (2/5 components ≥90%)
- **Overall Quality:** 82.7% (average across all)
- **Passing Components:** MCP Framework (94.5%), Core Systems (95%+)

### Components Needing Attention
1. **unknown_component** - 82.6% (gap: 7.4%, priority: 22)
2. **general_tests** - 70.7% (gap: 19.3%, priority: 12)
3. **comprehensive_e2e** - 74.4% (gap: 15.6%, priority: 10)

### Recommended Next Steps
1. Improve unknown_component by 7.4% (highest priority, smallest gap)
2. Address general_tests failing patterns (19.3% gap)
3. Enhance comprehensive_e2e coverage (15.6% gap)

## 🔧 Integration Points

### KPITrustMonitor
```python
# Record quality metric
await kpi_monitor.record_metric(
    name="test_quality_score",
    value=score * 100,
    component_id=component_id,
    threshold_warning=70,
    threshold_critical=50
)

# Update trust score
await kpi_monitor.update_trust_score(
    component_id=component_id,
    performance_score=trust_adjusted_score,
    confidence=min(1.0, total_tests / 10.0)
)
```

### EventBus/TriggerMesh
```python
# Publish quality events
await event_publisher(
    "test_quality.healing_required",
    {
        "component_id": component_id,
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }
)
```

### Self-Healing Loops
- **AVN Core:** Handles critical escalations
- **Learning Kernel:** Processes degraded components
- **Memory Orchestrator:** Coordinates healing actions

## 💡 Design Philosophy

### Principles
1. **Quality Over Quantity:** 90% threshold ensures meaningful pass metrics
2. **Adaptive Intelligence:** KPI integration provides historical context
3. **Self-Healing:** Automatic triggers reduce manual intervention
4. **Clear Communication:** Humans understand "2 of 5 passing" better than "76.7%"
5. **Incremental Progress:** Components improve gradually, count when ready

### Benefits
- ✅ Predictable progress tracking
- ✅ Automatic remediation triggering
- ✅ Clear visibility into component health
- ✅ Integration with existing Grace infrastructure
- ✅ No confusing partial percentages in final metrics

## 📝 File Structure

```
grace/testing/
├── __init__.py                    # Module exports
├── test_quality_monitor.py        # Core quality monitoring logic
└── pytest_quality_plugin.py       # Pytest integration

scripts/
└── view_quality_report.py         # CLI report viewer

test_reports/
└── quality_report_*.json          # Generated quality reports

conftest.py                         # Pytest configuration (enables plugin)
```

## 🎯 Future Enhancements

### Dynamic Thresholds
- Adjust threshold based on system health (85-95% range)
- Critical components require 95%, low-priority 85%
- Mission-based priority weighting

### Visualization
- Color-coded component dashboard
- Trend graphs showing quality over time
- Real-time quality monitoring UI

### Advanced Analytics
- Time-based weighting (recent > historical)
- Confidence intervals on quality scores
- Predictive quality forecasting

## 📚 Related Documentation

- KPITrustMonitor: `grace/core/kpi_trust_monitor.py`
- EventBus: `grace/core/event_bus.py`
- Memory Orchestrator: `grace/immune/avn_core.py`
- Learning Kernel: `grace/learning_kernel/`
- Test Status Report: `TEST_STATUS_REPORT.md`

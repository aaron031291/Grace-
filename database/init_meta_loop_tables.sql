-- ============================================================================
-- Grace Meta-Loop System - Database Tables
-- ============================================================================
-- Complete database schema for Grace's bi-directional cognitive architecture
-- Implements OODA loops (Observe, Orient, Decide, Act) with full auditability
-- Non-negotiable tracking of success, failure, and evolution
-- ============================================================================

-- ============================================================================
-- META-LOOP CORE TABLES
-- ============================================================================

-- 🔁 1. OBSERVATION LOOP (O-Loop) - Capture all sensory input
-- ============================================================================

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    observation_type TEXT NOT NULL, -- event, telemetry, command, anomaly, log
    source_module TEXT NOT NULL, -- Which kernel/component observed this
    observation_data TEXT NOT NULL, -- JSON: raw observation payload
    context TEXT NOT NULL, -- JSON: environmental context
    credibility_score REAL NOT NULL DEFAULT 1.0, -- 0.0-1.0, how trustworthy
    novelty_score REAL NOT NULL DEFAULT 0.5, -- 0.0-1.0, how new/unexpected
    priority TEXT NOT NULL DEFAULT 'normal', -- low, normal, high, critical
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    linked_to_knowledge BOOLEAN NOT NULL DEFAULT FALSE, -- Sent to Knowledge Core
    metadata TEXT -- JSON: additional context
);

CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(observation_type);
CREATE INDEX IF NOT EXISTS idx_observations_source ON observations(source_module);
CREATE INDEX IF NOT EXISTS idx_observations_processed ON observations(processed);
CREATE INDEX IF NOT EXISTS idx_observations_observed_at ON observations(observed_at);
CREATE INDEX IF NOT EXISTS idx_observations_novelty ON observations(novelty_score DESC);

-- Observation patterns (aggregated insights)
CREATE TABLE IF NOT EXISTS observation_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL, -- trend, anomaly, correlation, cycle
    observation_ids TEXT NOT NULL, -- JSON array of related observation IDs
    pattern_description TEXT NOT NULL,
    confidence REAL NOT NULL, -- 0.0-1.0
    frequency TEXT NOT NULL, -- once, rare, periodic, frequent
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    impact_assessment TEXT -- JSON: potential impacts
);

CREATE INDEX IF NOT EXISTS idx_observation_patterns_type ON observation_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_observation_patterns_confidence ON observation_patterns(confidence DESC);

-- ============================================================================
-- 🧭 2. ORIENTATION LOOP (R-Loop) - Analyze context and meaning
-- ============================================================================

CREATE TABLE IF NOT EXISTS orientations (
    orientation_id TEXT PRIMARY KEY,
    observation_ids TEXT NOT NULL, -- JSON array: which observations triggered this
    context_analysis TEXT NOT NULL, -- JSON: contextual interpretation
    correlations_found TEXT, -- JSON: relationships discovered
    contradictions_found TEXT, -- JSON: conflicts detected
    trust_weights TEXT NOT NULL, -- JSON: trust scores for each data source
    relevance_score REAL NOT NULL, -- 0.0-1.0, how relevant to current goals
    uncertainty_level REAL NOT NULL DEFAULT 0.0, -- 0.0-1.0, epistemic uncertainty
    orientation_complete BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    triggered_decision BOOLEAN NOT NULL DEFAULT FALSE -- Sent to Decision Loop
);

CREATE INDEX IF NOT EXISTS idx_orientations_complete ON orientations(orientation_complete);
CREATE INDEX IF NOT EXISTS idx_orientations_relevance ON orientations(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_orientations_created ON orientations(created_at);

-- Context drift detection
CREATE TABLE IF NOT EXISTS context_drifts (
    drift_id TEXT PRIMARY KEY,
    drift_type TEXT NOT NULL, -- data_drift, concept_drift, logic_drift
    old_context TEXT NOT NULL, -- JSON: previous understanding
    new_context TEXT NOT NULL, -- JSON: current understanding
    drift_magnitude REAL NOT NULL, -- 0.0-1.0, severity
    detected_by_orientation_id TEXT NOT NULL,
    requires_reorientation BOOLEAN NOT NULL DEFAULT TRUE,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    FOREIGN KEY (detected_by_orientation_id) REFERENCES orientations(orientation_id)
);

CREATE INDEX IF NOT EXISTS idx_context_drifts_type ON context_drifts(drift_type);
CREATE INDEX IF NOT EXISTS idx_context_drifts_magnitude ON context_drifts(drift_magnitude DESC);

-- ============================================================================
-- ⚖️ 3. DECISION LOOP (D-Loop) - Select optimal action
-- ============================================================================

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    orientation_id TEXT NOT NULL, -- Which orientation triggered this
    decision_type TEXT NOT NULL, -- action, escalation, delegation, defer, reject
    available_options TEXT NOT NULL, -- JSON array of possible actions
    selected_option TEXT NOT NULL, -- JSON: chosen action with parameters
    selection_rationale TEXT NOT NULL, -- Why this option was chosen
    confidence REAL NOT NULL, -- 0.0-1.0
    risk_assessment TEXT NOT NULL, -- JSON: identified risks
    ethical_check_passed BOOLEAN NOT NULL DEFAULT TRUE,
    trust_score REAL NOT NULL, -- Current trust level at decision time
    governance_approval_required BOOLEAN NOT NULL DEFAULT FALSE,
    governance_approved BOOLEAN,
    decided_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (orientation_id) REFERENCES orientations(orientation_id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_decisions_executed ON decisions(executed);
CREATE INDEX IF NOT EXISTS idx_decisions_confidence ON decisions(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON decisions(decided_at);

-- Decision alternatives (options not chosen - for learning)
CREATE TABLE IF NOT EXISTS decision_alternatives (
    alternative_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    option_description TEXT NOT NULL, -- JSON: the alternative action
    expected_outcome TEXT NOT NULL, -- JSON: predicted results
    rejection_reason TEXT NOT NULL,
    would_have_succeeded BOOLEAN, -- Counterfactual analysis (filled later)
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE INDEX IF NOT EXISTS idx_decision_alternatives_decision ON decision_alternatives(decision_id);

-- ============================================================================
-- ⚙️ 4. ACTION LOOP (A-Loop) - Execute operations
-- ============================================================================

CREATE TABLE IF NOT EXISTS actions (
    action_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    action_type TEXT NOT NULL, -- code_gen, data_update, env_change, api_call, query
    action_payload TEXT NOT NULL, -- JSON: full action specification
    execution_mode TEXT NOT NULL DEFAULT 'sandboxed', -- sandboxed, production
    executor_module TEXT NOT NULL, -- Which component executes this
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, succeeded, failed, rolled_back
    exit_code INTEGER,
    stdout_log TEXT, -- Execution output
    stderr_log TEXT, -- Error output
    side_effects TEXT, -- JSON: observed side effects
    rollback_possible BOOLEAN NOT NULL DEFAULT TRUE,
    rolled_back BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE INDEX IF NOT EXISTS idx_actions_decision ON actions(decision_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_started ON actions(started_at);
CREATE INDEX IF NOT EXISTS idx_actions_executor ON actions(executor_module);

-- Action lineage (for tracing causality)
CREATE TABLE IF NOT EXISTS action_lineage (
    lineage_id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL,
    parent_action_id TEXT, -- If this action spawned another
    depth INTEGER NOT NULL DEFAULT 0, -- Nesting level
    lineage_path TEXT NOT NULL, -- JSON array: full chain from root
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (action_id) REFERENCES actions(action_id),
    FOREIGN KEY (parent_action_id) REFERENCES actions(action_id)
);

CREATE INDEX IF NOT EXISTS idx_action_lineage_action ON action_lineage(action_id);
CREATE INDEX IF NOT EXISTS idx_action_lineage_parent ON action_lineage(parent_action_id);

-- ============================================================================
-- 🧮 5. EVALUATION LOOP (E-Loop) - Audit outcomes
-- ============================================================================

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL,
    intended_outcome TEXT NOT NULL, -- JSON: what was expected
    actual_outcome TEXT NOT NULL, -- JSON: what actually happened
    success BOOLEAN NOT NULL,
    performance_metrics TEXT NOT NULL, -- JSON: latency, accuracy, etc.
    side_effects_identified TEXT, -- JSON: unexpected consequences
    error_analysis TEXT, -- JSON: root cause if failed
    lessons_learned TEXT, -- JSON: insights from this execution
    confidence_adjustment REAL NOT NULL DEFAULT 0.0, -- +/- adjustment to trust
    evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sent_to_reflection BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (action_id) REFERENCES actions(action_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_action ON evaluations(action_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_success ON evaluations(success);
CREATE INDEX IF NOT EXISTS idx_evaluations_evaluated_at ON evaluations(evaluated_at);

-- Success/Failure patterns (for learning)
CREATE TABLE IF NOT EXISTS outcome_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL, -- recurring_success, recurring_failure, edge_case
    action_type TEXT NOT NULL,
    conditions TEXT NOT NULL, -- JSON: when does this pattern occur
    outcome TEXT NOT NULL, -- JSON: typical result
    frequency INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL, -- 0.0-1.0
    first_observed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_observed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actionable_insight TEXT -- What to do about this pattern
);

CREATE INDEX IF NOT EXISTS idx_outcome_patterns_type ON outcome_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_outcome_patterns_action ON outcome_patterns(action_type);
CREATE INDEX IF NOT EXISTS idx_outcome_patterns_confidence ON outcome_patterns(confidence DESC);

-- ============================================================================
-- 🪞 6. REFLECTION LOOP (F-Loop) - Pattern analysis over time
-- ============================================================================

CREATE TABLE IF NOT EXISTS reflections (
    reflection_id TEXT PRIMARY KEY,
    reflection_type TEXT NOT NULL, -- performance, error, trend, anomaly
    time_window_start TIMESTAMP NOT NULL,
    time_window_end TIMESTAMP NOT NULL,
    evaluation_ids TEXT NOT NULL, -- JSON array: which evaluations analyzed
    aggregated_metrics TEXT NOT NULL, -- JSON: summary statistics
    patterns_identified TEXT NOT NULL, -- JSON: discovered patterns
    trends_detected TEXT, -- JSON: upward/downward trends
    anomalies_detected TEXT, -- JSON: statistical outliers
    learning_insights TEXT NOT NULL, -- JSON: what was learned
    recommended_adjustments TEXT, -- JSON: proposed improvements
    reflected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    applied_to_evolution BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_reflections_reflected_at ON reflections(reflected_at);
CREATE INDEX IF NOT EXISTS idx_reflections_applied ON reflections(applied_to_evolution);

-- Learning heuristics (updated by reflection)
CREATE TABLE IF NOT EXISTS learning_heuristics (
    heuristic_id TEXT PRIMARY KEY,
    heuristic_name TEXT NOT NULL,
    heuristic_rule TEXT NOT NULL, -- JSON: the rule specification
    domain TEXT NOT NULL, -- Which area this applies to
    confidence REAL NOT NULL DEFAULT 0.5, -- 0.0-1.0
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    last_updated_by_reflection_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (last_updated_by_reflection_id) REFERENCES reflections(reflection_id)
);

CREATE INDEX IF NOT EXISTS idx_learning_heuristics_domain ON learning_heuristics(domain);
CREATE INDEX IF NOT EXISTS idx_learning_heuristics_active ON learning_heuristics(active);
CREATE INDEX IF NOT EXISTS idx_learning_heuristics_success_rate ON learning_heuristics(success_rate DESC);

-- ============================================================================
-- 🌱 7. EVOLUTION LOOP (V-Loop) - Systemic improvement
-- ============================================================================

CREATE TABLE IF NOT EXISTS evolution_proposals (
    proposal_id TEXT PRIMARY KEY,
    proposal_type TEXT NOT NULL, -- architecture_refactor, algorithm_swap, module_creation, parameter_tuning
    triggered_by_reflection_id TEXT,
    current_state TEXT NOT NULL, -- JSON: current system configuration
    proposed_change TEXT NOT NULL, -- JSON: detailed change specification
    expected_improvement TEXT NOT NULL, -- JSON: predicted benefits
    risk_analysis TEXT NOT NULL, -- JSON: identified risks
    governance_approval_required BOOLEAN NOT NULL DEFAULT TRUE,
    trust_threshold_required REAL NOT NULL DEFAULT 0.9,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, approved, rejected, testing, deployed, rolled_back
    FOREIGN KEY (triggered_by_reflection_id) REFERENCES reflections(reflection_id)
);

CREATE INDEX IF NOT EXISTS idx_evolution_proposals_type ON evolution_proposals(proposal_type);
CREATE INDEX IF NOT EXISTS idx_evolution_proposals_status ON evolution_proposals(status);
CREATE INDEX IF NOT EXISTS idx_evolution_proposals_created ON evolution_proposals(created_at);

-- Evolution experiments (sandbox testing)
CREATE TABLE IF NOT EXISTS evolution_experiments (
    experiment_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    sandbox_environment TEXT NOT NULL, -- Clone environment identifier
    experiment_config TEXT NOT NULL, -- JSON: test configuration
    baseline_metrics TEXT NOT NULL, -- JSON: metrics before change
    experiment_metrics TEXT, -- JSON: metrics after change
    improvement_achieved REAL, -- Percentage improvement
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'running', -- running, succeeded, failed
    recommendation TEXT, -- deploy, reject, iterate
    FOREIGN KEY (proposal_id) REFERENCES evolution_proposals(proposal_id)
);

CREATE INDEX IF NOT EXISTS idx_evolution_experiments_proposal ON evolution_experiments(proposal_id);
CREATE INDEX IF NOT EXISTS idx_evolution_experiments_status ON evolution_experiments(status);

-- Evolution deployments (actual upgrades)
CREATE TABLE IF NOT EXISTS evolution_deployments (
    deployment_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    pre_deployment_snapshot_id TEXT NOT NULL, -- For rollback
    deployment_strategy TEXT NOT NULL, -- canary, blue_green, rolling, immediate
    deployed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'deployed', -- deployed, validating, validated, rolled_back
    validation_metrics TEXT, -- JSON: post-deployment performance
    rollback_reason TEXT,
    FOREIGN KEY (proposal_id) REFERENCES evolution_proposals(proposal_id),
    FOREIGN KEY (experiment_id) REFERENCES evolution_experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_evolution_deployments_proposal ON evolution_deployments(proposal_id);
CREATE INDEX IF NOT EXISTS idx_evolution_deployments_status ON evolution_deployments(status);

-- ============================================================================
-- 🧠 8. TRUST & ETHICS LOOP (T-Loop) - Continuous compliance monitoring
-- ============================================================================

CREATE TABLE IF NOT EXISTS trust_measurements (
    measurement_id TEXT PRIMARY KEY,
    measurement_type TEXT NOT NULL, -- behavior, performance, alignment, safety
    component TEXT NOT NULL, -- Which module is being measured
    kpi_name TEXT NOT NULL, -- Which KPI is being checked
    target_value REAL NOT NULL,
    actual_value REAL NOT NULL,
    deviation REAL NOT NULL, -- How far from target
    trust_impact REAL NOT NULL, -- +/- adjustment to trust score
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    violations_detected TEXT, -- JSON: specific violations
    corrective_actions_needed TEXT -- JSON: required interventions
);

CREATE INDEX IF NOT EXISTS idx_trust_measurements_component ON trust_measurements(component);
CREATE INDEX IF NOT EXISTS idx_trust_measurements_measured_at ON trust_measurements(measured_at);
CREATE INDEX IF NOT EXISTS idx_trust_measurements_impact ON trust_measurements(trust_impact);

-- Trust score history (per component)
CREATE TABLE IF NOT EXISTS trust_scores (
    score_id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    trust_score REAL NOT NULL, -- 0.0-1.0
    previous_score REAL NOT NULL,
    change_reason TEXT NOT NULL, -- Why score changed
    measurement_id TEXT, -- Which measurement caused this
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    threshold_breached BOOLEAN NOT NULL DEFAULT FALSE,
    action_taken TEXT, -- throttle, gate, alert, halt
    FOREIGN KEY (measurement_id) REFERENCES trust_measurements(measurement_id)
);

CREATE INDEX IF NOT EXISTS idx_trust_scores_component ON trust_scores(component);
CREATE INDEX IF NOT EXISTS idx_trust_scores_timestamp ON trust_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_trust_scores_breached ON trust_scores(threshold_breached);

-- Ethics violations (non-negotiable tracking)
CREATE TABLE IF NOT EXISTS ethics_violations (
    violation_id TEXT PRIMARY KEY,
    violation_type TEXT NOT NULL, -- safety, fairness, transparency, privacy, consent
    severity TEXT NOT NULL, -- warning, error, critical
    component TEXT NOT NULL,
    action_id TEXT, -- Which action violated
    decision_id TEXT, -- Which decision violated
    violation_details TEXT NOT NULL, -- JSON: specifics
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolution TEXT, -- rollback, patch, policy_update, manual_intervention
    prevented_harm BOOLEAN NOT NULL DEFAULT FALSE, -- Was action stopped in time?
    FOREIGN KEY (action_id) REFERENCES actions(action_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE INDEX IF NOT EXISTS idx_ethics_violations_type ON ethics_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_ethics_violations_severity ON ethics_violations(severity);
CREATE INDEX IF NOT EXISTS idx_ethics_violations_detected ON ethics_violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_ethics_violations_resolved ON ethics_violations(resolved_at);

-- ============================================================================
-- 🔄 9. KNOWLEDGE FUSION LOOP (K-Loop) - Continuous learning
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_ingestion (
    ingestion_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL, -- web, pdf, chat, log, api, human
    source_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- For deduplication
    raw_content TEXT NOT NULL, -- Original content
    normalized_content TEXT, -- Processed content
    credibility_score REAL NOT NULL DEFAULT 0.5, -- 0.0-1.0
    provenance TEXT NOT NULL, -- JSON: source chain
    integrity_check_passed BOOLEAN NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    linked_to_graph BOOLEAN NOT NULL DEFAULT FALSE -- Added to Knowledge Core
);

CREATE INDEX IF NOT EXISTS idx_knowledge_ingestion_source ON knowledge_ingestion(source_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_ingestion_hash ON knowledge_ingestion(content_hash);
CREATE INDEX IF NOT EXISTS idx_knowledge_ingestion_credibility ON knowledge_ingestion(credibility_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_ingestion_processed ON knowledge_ingestion(processed);

-- Knowledge graph updates (semantic relationships)
CREATE TABLE IF NOT EXISTS knowledge_graph_updates (
    update_id TEXT PRIMARY KEY,
    ingestion_id TEXT NOT NULL,
    update_type TEXT NOT NULL, -- add_node, add_edge, update_node, delete_node
    entity_id TEXT NOT NULL, -- Which knowledge entity
    entity_type TEXT NOT NULL, -- concept, fact, relationship, rule
    update_payload TEXT NOT NULL, -- JSON: the actual update
    semantic_embedding TEXT, -- Vector embedding for similarity
    confidence REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ingestion_id) REFERENCES knowledge_ingestion(ingestion_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_updates_ingestion ON knowledge_graph_updates(ingestion_id);
CREATE INDEX IF NOT EXISTS idx_kg_updates_entity ON knowledge_graph_updates(entity_id);
CREATE INDEX IF NOT EXISTS idx_kg_updates_type ON knowledge_graph_updates(update_type);

-- ============================================================================
-- META-LOOP ORCHESTRATION
-- ============================================================================

-- Loop execution history (tracking all loop runs)
CREATE TABLE IF NOT EXISTS meta_loop_executions (
    execution_id TEXT PRIMARY KEY,
    loop_type TEXT NOT NULL, -- O, R, D, A, E, F, V, T, K
    loop_name TEXT NOT NULL, -- Full loop name
    triggered_by TEXT, -- What triggered this loop execution
    input_data TEXT NOT NULL, -- JSON: loop inputs
    output_data TEXT, -- JSON: loop outputs
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'running', -- running, completed, failed, timeout
    error_message TEXT,
    next_loop_triggered TEXT -- Which loop(s) this triggered
);

CREATE INDEX IF NOT EXISTS idx_meta_loop_executions_type ON meta_loop_executions(loop_type);
CREATE INDEX IF NOT EXISTS idx_meta_loop_executions_started ON meta_loop_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_meta_loop_executions_status ON meta_loop_executions(status);

-- Loop dependencies (which loops trigger which)
CREATE TABLE IF NOT EXISTS meta_loop_dependencies (
    dependency_id TEXT PRIMARY KEY,
    source_loop TEXT NOT NULL, -- Which loop
    target_loop TEXT NOT NULL, -- Triggers which loop
    condition TEXT NOT NULL, -- JSON: when does trigger occur
    priority INTEGER NOT NULL DEFAULT 5,
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_meta_loop_deps_source ON meta_loop_dependencies(source_loop);
CREATE INDEX IF NOT EXISTS idx_meta_loop_deps_target ON meta_loop_dependencies(target_loop);

-- Loop performance metrics
CREATE TABLE IF NOT EXISTS meta_loop_metrics (
    metric_id TEXT PRIMARY KEY,
    loop_type TEXT NOT NULL,
    metric_name TEXT NOT NULL, -- avg_latency, success_rate, throughput
    metric_value REAL NOT NULL,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    calculated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_meta_loop_metrics_loop ON meta_loop_metrics(loop_type);
CREATE INDEX IF NOT EXISTS idx_meta_loop_metrics_calculated ON meta_loop_metrics(calculated_at);

-- ============================================================================
-- VIEWS FOR LOOP ANALYSIS
-- ============================================================================

-- Current system health across all loops
CREATE VIEW IF NOT EXISTS meta_loop_health AS
SELECT 
    loop_type,
    COUNT(*) as total_executions,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(duration_ms) as avg_duration_ms,
    MAX(started_at) as last_execution
FROM meta_loop_executions
WHERE started_at >= datetime('now', '-24 hours')
GROUP BY loop_type;

-- OODA cycle completeness (are all loops running?)
CREATE VIEW IF NOT EXISTS ooda_cycle_status AS
SELECT 
    o.observation_id,
    o.observed_at,
    r.orientation_id,
    d.decision_id,
    a.action_id,
    e.evaluation_id,
    CASE 
        WHEN e.evaluation_id IS NOT NULL THEN 'complete'
        WHEN a.action_id IS NOT NULL THEN 'executing'
        WHEN d.decision_id IS NOT NULL THEN 'decided'
        WHEN r.orientation_id IS NOT NULL THEN 'oriented'
        ELSE 'observed'
    END as cycle_stage
FROM observations o
LEFT JOIN orientations r ON o.observation_id IN (SELECT json_extract(value, '$') FROM json_each(r.observation_ids))
LEFT JOIN decisions d ON r.orientation_id = d.orientation_id
LEFT JOIN actions a ON d.decision_id = a.decision_id
LEFT JOIN evaluations e ON a.action_id = e.action_id
WHERE o.observed_at >= datetime('now', '-1 hour')
ORDER BY o.observed_at DESC;

-- Trust score trends
CREATE VIEW IF NOT EXISTS trust_trends AS
SELECT 
    component,
    trust_score,
    previous_score,
    (trust_score - previous_score) as change,
    change_reason,
    timestamp
FROM trust_scores
WHERE timestamp >= datetime('now', '-7 days')
ORDER BY timestamp DESC;



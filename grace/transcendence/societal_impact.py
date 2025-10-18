"""
Societal Impact Evaluator - Policy simulation and impact assessment
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicySimulation:
    """Policy simulation result"""
    policy_id: str
    policy_description: str
    affected_populations: Dict[str, int]
    projected_outcomes: Dict[str, float]
    risks: List[Dict[str, Any]]
    benefits: List[Dict[str, Any]]
    confidence: float
    time_horizon: str
    metadata: Dict[str, Any]


class SocietalImpactEvaluator:
    """
    Simulates and evaluates societal impact of policies
    
    Features:
    - Multi-stakeholder analysis
    - Long-term projection
    - Risk assessment
    - Ethical evaluation
    """
    
    def __init__(self):
        self.simulations: List[PolicySimulation] = []
        self.stakeholder_models: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SocietalImpactEvaluator initialized")
    
    def simulate_policy(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any],
        time_horizon: str = "1_year"
    ) -> PolicySimulation:
        """
        Simulate impact of a policy
        
        Args:
            policy: Policy details
            context: Current societal context
            time_horizon: Projection timeframe
            
        Returns:
            Policy simulation results
        """
        policy_id = policy.get("id", f"policy_{len(self.simulations)}")
        
        # Identify affected populations
        affected = self._identify_affected_populations(policy, context)
        
        # Project outcomes
        outcomes = self._project_outcomes(policy, context, time_horizon)
        
        # Assess risks
        risks = self._assess_risks(policy, context, outcomes)
        
        # Identify benefits
        benefits = self._identify_benefits(policy, context, outcomes)
        
        # Calculate confidence
        confidence = self._calculate_simulation_confidence(policy, context, outcomes)
        
        simulation = PolicySimulation(
            policy_id=policy_id,
            policy_description=policy.get("description", ""),
            affected_populations=affected,
            projected_outcomes=outcomes,
            risks=risks,
            benefits=benefits,
            confidence=confidence,
            time_horizon=time_horizon,
            metadata={
                "simulated_at": datetime.now(timezone.utc).isoformat(),
                "context_snapshot": context
            }
        )
        
        self.simulations.append(simulation)
        
        logger.info(f"Simulated policy {policy_id}: confidence={confidence:.3f}")
        return simulation
    
    def _identify_affected_populations(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, int]:
        """Identify populations affected by policy"""
        affected = {}
        
        # Extract target groups from policy
        target_groups = policy.get("target_groups", [])
        total_population = context.get("total_population", 1000000)
        
        if not target_groups:
            # Default: assume affects general population
            affected["general_population"] = int(total_population * 0.3)
        else:
            for group in target_groups:
                # Estimate group size
                group_fraction = context.get(f"{group}_fraction", 0.1)
                affected[group] = int(total_population * group_fraction)
        
        return affected
    
    def _project_outcomes(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any],
        time_horizon: str
    ) -> Dict[str, float]:
        """Project policy outcomes"""
        outcomes = {}
        
        # Time multiplier
        time_multipliers = {
            "1_month": 0.1,
            "3_months": 0.25,
            "6_months": 0.5,
            "1_year": 1.0,
            "5_years": 3.0,
            "10_years": 5.0
        }
        
        time_mult = time_multipliers.get(time_horizon, 1.0)
        
        # Economic impact
        economic_effect = policy.get("economic_effect", 0.0)
        outcomes["economic_impact"] = economic_effect * time_mult
        
        # Social welfare
        welfare_effect = policy.get("welfare_effect", 0.0)
        outcomes["social_welfare_change"] = welfare_effect * time_mult
        
        # Environmental impact
        env_effect = policy.get("environmental_effect", 0.0)
        outcomes["environmental_impact"] = env_effect * time_mult
        
        # Employment
        employment_effect = policy.get("employment_effect", 0.0)
        outcomes["employment_change"] = employment_effect * time_mult
        
        # Health outcomes
        health_effect = policy.get("health_effect", 0.0)
        outcomes["health_impact"] = health_effect * time_mult
        
        # Add uncertainty
        for key in outcomes:
            noise = np.random.normal(0, 0.1)
            outcomes[key] *= (1 + noise)
        
        return outcomes
    
    def _assess_risks(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any],
        outcomes: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Assess risks associated with policy"""
        risks = []
        
        # Economic risks
        if outcomes.get("economic_impact", 0) < -0.5:
            risks.append({
                "type": "economic",
                "severity": "high",
                "description": "Significant negative economic impact",
                "likelihood": 0.7,
                "mitigation": "Implement gradual rollout with economic safeguards"
            })
        
        # Social disruption risks
        affected_count = sum(policy.get("affected_populations", {}).values())
        if affected_count > 100000:
            risks.append({
                "type": "social",
                "severity": "medium",
                "description": "Large-scale social disruption possible",
                "likelihood": 0.5,
                "mitigation": "Extensive stakeholder consultation and phased implementation"
            })
        
        # Unintended consequences
        if abs(outcomes.get("environmental_impact", 0)) > 1.0:
            risks.append({
                "type": "environmental",
                "severity": "medium",
                "description": "Potential environmental consequences",
                "likelihood": 0.4,
                "mitigation": "Environmental impact assessment required"
            })
        
        # Equity risks
        if policy.get("equity_impact", 0) < -0.3:
            risks.append({
                "type": "equity",
                "severity": "high",
                "description": "May exacerbate inequality",
                "likelihood": 0.6,
                "mitigation": "Add compensatory measures for affected groups"
            })
        
        return risks
    
    def _identify_benefits(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any],
        outcomes: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify benefits of policy"""
        benefits = []
        
        # Economic benefits
        if outcomes.get("economic_impact", 0) > 0.3:
            benefits.append({
                "type": "economic",
                "magnitude": "significant",
                "description": "Positive economic growth expected",
                "beneficiaries": "general_population"
            })
        
        # Health benefits
        if outcomes.get("health_impact", 0) > 0.2:
            benefits.append({
                "type": "health",
                "magnitude": "moderate",
                "description": "Improved health outcomes",
                "beneficiaries": "healthcare_recipients"
            })
        
        # Employment benefits
        if outcomes.get("employment_change", 0) > 0.1:
            benefits.append({
                "type": "employment",
                "magnitude": "moderate",
                "description": "Job creation expected",
                "beneficiaries": "workforce"
            })
        
        # Environmental benefits
        if outcomes.get("environmental_impact", 0) > 0.5:
            benefits.append({
                "type": "environmental",
                "magnitude": "significant",
                "description": "Environmental improvement",
                "beneficiaries": "future_generations"
            })
        
        return benefits
    
    def _calculate_simulation_confidence(
        self,
        policy: Dict[str, Any],
        context: Dict[str, Any],
        outcomes: Dict[str, float]
    ) -> float:
        """Calculate confidence in simulation"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on data availability
        if "historical_data" in context:
            confidence += 0.1
        
        # Adjust based on policy clarity
        if policy.get("well_defined", False):
            confidence += 0.1
        
        # Reduce for high uncertainty
        outcome_variance = np.var(list(outcomes.values()))
        confidence -= min(0.3, outcome_variance * 0.1)
        
        return max(0.0, min(1.0, confidence))
    
    def compare_policies(
        self,
        policies: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare multiple policies"""
        simulations = [
            self.simulate_policy(policy, context)
            for policy in policies
        ]
        
        # Rank by overall benefit
        rankings = []
        
        for sim in simulations:
            # Calculate net benefit score
            benefit_score = sum(
                1.0 if b["magnitude"] == "significant" else 0.5
                for b in sim.benefits
            )
            
            risk_score = sum(
                1.0 if r["severity"] == "high" else 0.5
                for r in sim.risks
            )
            
            net_score = benefit_score - risk_score * 0.7
            
            rankings.append({
                "policy_id": sim.policy_id,
                "net_score": net_score,
                "confidence": sim.confidence,
                "benefits": len(sim.benefits),
                "risks": len(sim.risks)
            })
        
        # Sort by net score
        rankings.sort(key=lambda x: x["net_score"], reverse=True)
        
        return {
            "rankings": rankings,
            "recommendation": rankings[0]["policy_id"] if rankings else None,
            "comparison_confidence": np.mean([r["confidence"] for r in rankings])
        }

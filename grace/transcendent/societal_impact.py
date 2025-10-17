"""
Societal Impact Evaluator - Ethics and policy foresight simulation
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ImpactDimension(Enum):
    """Dimensions of societal impact"""
    ECONOMIC = "economic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    ETHICAL = "ethical"
    POLITICAL = "political"
    TECHNOLOGICAL = "technological"
    CULTURAL = "cultural"


class RiskLevel(Enum):
    """Risk level classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StakeholderGroup:
    """Represents a stakeholder group"""
    id: str
    name: str
    size: int
    influence_score: float
    vulnerability_score: float
    interests: List[str]
    concerns: List[str]


@dataclass
class ImpactAssessment:
    """Assessment of societal impact"""
    id: str
    action_or_policy: str
    dimensions: Dict[ImpactDimension, float]  # -1 to 1 scale
    affected_stakeholders: List[str]
    timeframe: str
    confidence: float
    risks: List[Dict[str, Any]]
    benefits: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicalDilemma:
    """Represents an ethical dilemma"""
    id: str
    description: str
    conflicting_values: List[str]
    stakeholder_positions: Dict[str, str]
    severity: RiskLevel
    resolution_options: List[Dict[str, Any]]
    recommended_approach: Optional[str] = None


@dataclass
class PolicySimulation:
    """Policy implementation simulation"""
    id: str
    policy_description: str
    implementation_timeline: List[Dict[str, Any]]
    projected_outcomes: Dict[str, Any]
    sensitivity_analysis: Dict[str, float]
    unintended_consequences: List[Dict[str, Any]]
    success_probability: float


class SocietalImpactEvaluator:
    """
    Evaluates societal impact of decisions, policies, and technologies
    Provides ethical analysis and policy foresight
    """
    
    def __init__(self):
        self.stakeholders: Dict[str, StakeholderGroup] = {}
        self.assessments: Dict[str, ImpactAssessment] = {}
        self.dilemmas: Dict[str, EthicalDilemma] = {}
        self.simulations: Dict[str, PolicySimulation] = {}
        self.ethical_frameworks: List[str] = [
            "utilitarianism",
            "deontology",
            "virtue_ethics",
            "care_ethics",
            "justice_as_fairness"
        ]
        logger.info("SocietalImpactEvaluator initialized")
    
    def register_stakeholder(
        self,
        name: str,
        size: int,
        influence: float,
        vulnerability: float,
        interests: List[str],
        concerns: List[str]
    ) -> StakeholderGroup:
        """Register a stakeholder group"""
        stakeholder = StakeholderGroup(
            id=f"stake_{len(self.stakeholders)}",
            name=name,
            size=size,
            influence_score=influence,
            vulnerability_score=vulnerability,
            interests=interests,
            concerns=concerns
        )
        
        self.stakeholders[stakeholder.id] = stakeholder
        logger.info(f"Registered stakeholder: {name}")
        
        return stakeholder
    
    def assess_impact(
        self,
        action_description: str,
        context: Dict[str, Any],
        timeframe: str = "medium_term"
    ) -> ImpactAssessment:
        """Assess societal impact of an action or policy"""
        
        # Analyze impact across dimensions
        dimensions = {}
        for dim in ImpactDimension:
            dimensions[dim] = self._calculate_dimensional_impact(
                action_description,
                dim,
                context
            )
        
        # Identify affected stakeholders
        affected = self._identify_affected_stakeholders(action_description, context)
        
        # Assess risks and benefits
        risks = self._identify_risks(action_description, dimensions, affected)
        benefits = self._identify_benefits(action_description, dimensions, affected)
        
        # Generate mitigation strategies
        mitigations = self._generate_mitigations(risks)
        
        # Calculate confidence
        confidence = self._calculate_assessment_confidence(context, len(affected))
        
        assessment = ImpactAssessment(
            id=f"impact_{len(self.assessments)}",
            action_or_policy=action_description,
            dimensions=dimensions,
            affected_stakeholders=[s.id for s in affected],
            timeframe=timeframe,
            confidence=confidence,
            risks=risks,
            benefits=benefits,
            mitigation_strategies=mitigations,
            metadata={'context': context, 'assessed_at': datetime.now().isoformat()}
        )
        
        self.assessments[assessment.id] = assessment
        logger.info(f"Completed impact assessment: {assessment.id}")
        
        return assessment
    
    def analyze_ethical_dilemma(
        self,
        situation: str,
        conflicting_values: List[str],
        stakeholder_ids: List[str]
    ) -> EthicalDilemma:
        """Analyze ethical dilemma from multiple frameworks"""
        
        # Gather stakeholder positions
        positions = {}
        for sid in stakeholder_ids:
            if sid in self.stakeholders:
                positions[sid] = self._determine_stakeholder_position(
                    self.stakeholders[sid],
                    situation,
                    conflicting_values
                )
        
        # Generate resolution options from each ethical framework
        resolution_options = []
        for framework in self.ethical_frameworks:
            option = self._apply_ethical_framework(
                framework,
                situation,
                conflicting_values,
                positions
            )
            resolution_options.append(option)
        
        # Determine severity
        severity = self._assess_dilemma_severity(
            conflicting_values,
            stakeholder_ids
        )
        
        # Recommend approach (balanced consideration)
        recommended = self._recommend_ethical_approach(resolution_options)
        
        dilemma = EthicalDilemma(
            id=f"dilemma_{len(self.dilemmas)}",
            description=situation,
            conflicting_values=conflicting_values,
            stakeholder_positions=positions,
            severity=severity,
            resolution_options=resolution_options,
            recommended_approach=recommended
        )
        
        self.dilemmas[dilemma.id] = dilemma
        logger.info(f"Analyzed ethical dilemma: {dilemma.id}")
        
        return dilemma
    
    def simulate_policy(
        self,
        policy: str,
        implementation_plan: Dict[str, Any],
        simulation_duration: int = 365  # days
    ) -> PolicySimulation:
        """Simulate policy implementation and outcomes"""
        
        # Create implementation timeline
        timeline = self._create_implementation_timeline(
            policy,
            implementation_plan,
            simulation_duration
        )
        
        # Project outcomes
        outcomes = self._project_policy_outcomes(
            policy,
            timeline,
            simulation_duration
        )
        
        # Sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(policy, implementation_plan)
        
        # Identify unintended consequences
        unintended = self._identify_unintended_consequences(
            policy,
            outcomes,
            timeline
        )
        
        # Calculate success probability
        success_prob = self._calculate_success_probability(
            outcomes,
            unintended,
            sensitivity
        )
        
        simulation = PolicySimulation(
            id=f"sim_{len(self.simulations)}",
            policy_description=policy,
            implementation_timeline=timeline,
            projected_outcomes=outcomes,
            sensitivity_analysis=sensitivity,
            unintended_consequences=unintended,
            success_probability=success_prob
        )
        
        self.simulations[simulation.id] = simulation
        logger.info(f"Completed policy simulation: {simulation.id}")
        
        return simulation
    
    def compare_alternatives(
        self,
        alternatives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare alternative actions/policies"""
        comparisons = []
        
        for alt in alternatives:
            # Assess each alternative
            assessment = self.assess_impact(
                alt['description'],
                alt.get('context', {}),
                alt.get('timeframe', 'medium_term')
            )
            
            # Calculate aggregate scores
            ethical_score = self._calculate_ethical_score(assessment)
            utilitarian_score = self._calculate_utilitarian_score(assessment)
            risk_score = self._calculate_risk_score(assessment)
            
            comparisons.append({
                'alternative': alt['description'],
                'assessment_id': assessment.id,
                'ethical_score': ethical_score,
                'utilitarian_score': utilitarian_score,
                'risk_score': risk_score,
                'net_impact': ethical_score * 0.4 + utilitarian_score * 0.4 - risk_score * 0.2
            })
        
        # Rank alternatives
        ranked = sorted(comparisons, key=lambda x: x['net_impact'], reverse=True)
        
        return {
            'comparisons': ranked,
            'recommended': ranked[0] if ranked else None,
            'trade_offs': self._identify_trade_offs(ranked)
        }
    
    def _calculate_dimensional_impact(
        self,
        action: str,
        dimension: ImpactDimension,
        context: Dict
    ) -> float:
        """Calculate impact on specific dimension (-1 to 1)"""
        # Simplified impact calculation
        keywords = {
            ImpactDimension.ECONOMIC: ['cost', 'profit', 'employment', 'growth'],
            ImpactDimension.SOCIAL: ['community', 'equality', 'access', 'wellbeing'],
            ImpactDimension.ENVIRONMENTAL: ['sustainability', 'pollution', 'climate', 'resources'],
            ImpactDimension.ETHICAL: ['fairness', 'rights', 'dignity', 'justice'],
            ImpactDimension.POLITICAL: ['governance', 'power', 'policy', 'regulation'],
            ImpactDimension.TECHNOLOGICAL: ['innovation', 'automation', 'digital', 'AI'],
            ImpactDimension.CULTURAL: ['tradition', 'values', 'diversity', 'heritage']
        }
        
        action_lower = action.lower()
        dim_keywords = keywords.get(dimension, [])
        
        # Simple keyword matching for impact
        positive_count = sum(1 for kw in dim_keywords if kw in action_lower)
        
        # Context-based adjustment
        context_modifier = context.get(f'{dimension.value}_modifier', 0)
        
        impact = (positive_count / max(len(dim_keywords), 1)) - 0.5 + context_modifier
        return max(-1.0, min(1.0, impact))
    
    def _identify_affected_stakeholders(
        self,
        action: str,
        context: Dict
    ) -> List[StakeholderGroup]:
        """Identify stakeholders affected by action"""
        affected = []
        
        action_lower = action.lower()
        for stakeholder in self.stakeholders.values():
            # Check if stakeholder interests/concerns align with action
            relevance = sum(
                1 for interest in stakeholder.interests + stakeholder.concerns
                if interest.lower() in action_lower
            )
            
            if relevance > 0:
                affected.append(stakeholder)
        
        return affected
    
    def _identify_risks(
        self,
        action: str,
        dimensions: Dict[ImpactDimension, float],
        stakeholders: List[StakeholderGroup]
    ) -> List[Dict[str, Any]]:
        """Identify risks"""
        risks = []
        
        # Negative dimensional impacts
        for dim, impact in dimensions.items():
            if impact < -0.3:
                risks.append({
                    'type': 'negative_impact',
                    'dimension': dim.value,
                    'severity': abs(impact),
                    'description': f"Negative {dim.value} impact"
                })
        
        # Vulnerable stakeholder risks
        for stakeholder in stakeholders:
            if stakeholder.vulnerability_score > 0.6:
                risks.append({
                    'type': 'stakeholder_vulnerability',
                    'stakeholder': stakeholder.name,
                    'severity': stakeholder.vulnerability_score,
                    'description': f"Risk to vulnerable group: {stakeholder.name}"
                })
        
        return risks
    
    def _identify_benefits(
        self,
        action: str,
        dimensions: Dict[ImpactDimension, float],
        stakeholders: List[StakeholderGroup]
    ) -> List[Dict[str, Any]]:
        """Identify benefits"""
        benefits = []
        
        for dim, impact in dimensions.items():
            if impact > 0.3:
                benefits.append({
                    'dimension': dim.value,
                    'magnitude': impact,
                    'description': f"Positive {dim.value} impact"
                })
        
        return benefits
    
    def _generate_mitigations(self, risks: List[Dict]) -> List[str]:
        """Generate mitigation strategies"""
        mitigations = []
        
        for risk in risks:
            if risk['type'] == 'negative_impact':
                mitigations.append(f"Implement safeguards for {risk['dimension']} concerns")
            elif risk['type'] == 'stakeholder_vulnerability':
                mitigations.append(f"Provide support and protection for {risk['stakeholder']}")
        
        return mitigations
    
    def _calculate_assessment_confidence(self, context: Dict, num_stakeholders: int) -> float:
        """Calculate confidence in assessment"""
        base_confidence = 0.5
        
        # More context increases confidence
        if len(context) > 5:
            base_confidence += 0.2
        
        # More stakeholders analyzed increases confidence
        if num_stakeholders > 3:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _determine_stakeholder_position(
        self,
        stakeholder: StakeholderGroup,
        situation: str,
        values: List[str]
    ) -> str:
        """Determine stakeholder position on dilemma"""
        # Align position with stakeholder interests
        for value in values:
            if value.lower() in [i.lower() for i in stakeholder.interests]:
                return f"Supports {value}"
        
        return "Neutral"
    
    def _apply_ethical_framework(
        self,
        framework: str,
        situation: str,
        values: List[str],
        positions: Dict
    ) -> Dict[str, Any]:
        """Apply ethical framework to situation"""
        frameworks_logic = {
            'utilitarianism': 'Maximize overall welfare and happiness',
            'deontology': 'Follow moral duties and rules regardless of outcomes',
            'virtue_ethics': 'Act according to virtuous character traits',
            'care_ethics': 'Prioritize relationships and compassion',
            'justice_as_fairness': 'Ensure fair distribution and equal opportunity'
        }
        
        return {
            'framework': framework,
            'principle': frameworks_logic.get(framework, 'Unknown'),
            'recommendation': f"From {framework}: consider {frameworks_logic.get(framework, '')}"
        }
    
    def _assess_dilemma_severity(self, values: List[str], stakeholders: List[str]) -> RiskLevel:
        """Assess severity of ethical dilemma"""
        if len(values) > 3 or len(stakeholders) > 5:
            return RiskLevel.HIGH
        elif len(values) > 2 or len(stakeholders) > 3:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _recommend_ethical_approach(self, options: List[Dict]) -> str:
        """Recommend balanced ethical approach"""
        # Recommend care ethics or justice for complex dilemmas
        for option in options:
            if option['framework'] in ['care_ethics', 'justice_as_fairness']:
                return option['recommendation']
        
        return options[0]['recommendation'] if options else "Seek further consultation"
    
    def _create_implementation_timeline(
        self,
        policy: str,
        plan: Dict,
        duration: int
    ) -> List[Dict[str, Any]]:
        """Create policy implementation timeline"""
        timeline = []
        
        phases = ['planning', 'pilot', 'rollout', 'evaluation']
        phase_duration = duration // len(phases)
        
        for i, phase in enumerate(phases):
            timeline.append({
                'phase': phase,
                'start_day': i * phase_duration,
                'end_day': (i + 1) * phase_duration,
                'activities': plan.get(f'{phase}_activities', []),
                'milestones': plan.get(f'{phase}_milestones', [])
            })
        
        return timeline
    
    def _project_policy_outcomes(
        self,
        policy: str,
        timeline: List[Dict],
        duration: int
    ) -> Dict[str, Any]:
        """Project policy outcomes"""
        return {
            'short_term': {'impact': 0.3, 'description': 'Initial implementation effects'},
            'medium_term': {'impact': 0.6, 'description': 'Policy taking effect'},
            'long_term': {'impact': 0.8, 'description': 'Full policy integration'},
            'affected_population': 1000000,  # Example
            'cost_benefit_ratio': 1.5
        }
    
    def _perform_sensitivity_analysis(self, policy: str, plan: Dict) -> Dict[str, float]:
        """Perform sensitivity analysis"""
        return {
            'economic_conditions': 0.4,
            'political_support': 0.6,
            'public_acceptance': 0.5,
            'resource_availability': 0.3,
            'technological_factors': 0.2
        }
    
    def _identify_unintended_consequences(
        self,
        policy: str,
        outcomes: Dict,
        timeline: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Identify potential unintended consequences"""
        return [
            {
                'consequence': 'Potential resource reallocation issues',
                'probability': 0.3,
                'severity': 'moderate',
                'timeframe': 'medium_term'
            }
        ]
    
    def _calculate_success_probability(
        self,
        outcomes: Dict,
        unintended: List[Dict],
        sensitivity: Dict
    ) -> float:
        """Calculate policy success probability"""
        base_prob = 0.7
        
        # Reduce by number of unintended consequences
        base_prob -= len(unintended) * 0.05
        
        # Adjust by sensitivity factors
        avg_sensitivity = sum(sensitivity.values()) / len(sensitivity)
        base_prob *= (1 - avg_sensitivity * 0.3)
        
        return max(0.1, min(0.95, base_prob))
    
    def _calculate_ethical_score(self, assessment: ImpactAssessment) -> float:
        """Calculate ethical alignment score"""
        ethical_impact = assessment.dimensions.get(ImpactDimension.ETHICAL, 0)
        social_impact = assessment.dimensions.get(ImpactDimension.SOCIAL, 0)
        
        return (ethical_impact + social_impact) / 2
    
    def _calculate_utilitarian_score(self, assessment: ImpactAssessment) -> float:
        """Calculate utilitarian (greatest good) score"""
        # Sum all positive impacts
        total_impact = sum(max(0, v) for v in assessment.dimensions.values())
        return total_impact / len(assessment.dimensions)
    
    def _calculate_risk_score(self, assessment: ImpactAssessment) -> float:
        """Calculate risk score"""
        if not assessment.risks:
            return 0.0
        
        avg_severity = sum(r.get('severity', 0.5) for r in assessment.risks) / len(assessment.risks)
        return avg_severity
    
    def _identify_trade_offs(self, ranked_alternatives: List[Dict]) -> List[Dict[str, Any]]:
        """Identify trade-offs between alternatives"""
        if len(ranked_alternatives) < 2:
            return []
        
        best = ranked_alternatives[0]
        second = ranked_alternatives[1]
        
        return [{
            'description': f"Top choice has higher net impact but consider trade-offs",
            'best_strength': 'ethical_score',
            'best_weakness': 'risk_score',
            'alternative_strength': second['alternative']
        }]
    
    def get_overall_report(self) -> Dict[str, Any]:
        """Generate overall societal impact report"""
        return {
            'total_assessments': len(self.assessments),
            'ethical_dilemmas_analyzed': len(self.dilemmas),
            'policy_simulations': len(self.simulations),
            'registered_stakeholders': len(self.stakeholders),
            'high_risk_assessments': sum(
                1 for a in self.assessments.values()
                if any(r.get('severity', 0) > 0.7 for r in a.risks)
            ),
            'frameworks_used': self.ethical_frameworks
        }

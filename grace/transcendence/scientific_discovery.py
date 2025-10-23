"""
Scientific Discovery Accelerator - Hypothesis generation from data
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Scientific hypothesis"""
    hypothesis_id: str
    statement: str
    confidence: float
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    testable: bool
    variables: List[str]
    predicted_outcomes: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScientificDiscoveryAccelerator:
    """
    Generates scientific hypotheses from data patterns
    
    Features:
    - Pattern recognition
    - Causal inference
    - Hypothesis generation
    - Experiment design
    """
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.hypotheses: List[Hypothesis] = []
        self.discovered_patterns: List[Dict[str, Any]] = []
        
        logger.info("ScientificDiscoveryAccelerator initialized")
    
    def analyze_data(
        self,
        data: List[Dict[str, Any]],
        target_variable: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze data to discover patterns
        
        Returns list of discovered patterns
        """
        if len(data) < 10:
            logger.warning("Insufficient data for pattern discovery")
            return []
        
        patterns = []
        
        # Extract variables
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        variables = list(all_keys)
        
        # Discover correlations
        correlations = self._find_correlations(data, variables)
        patterns.extend(correlations)
        
        # Discover trends
        trends = self._find_trends(data, variables)
        patterns.extend(trends)
        
        # Discover clusters
        clusters = self._find_clusters(data, variables)
        patterns.extend(clusters)
        
        # Store patterns
        self.discovered_patterns.extend(patterns)
        
        logger.info(f"Discovered {len(patterns)} patterns in data")
        return patterns
    
    def generate_hypotheses(
        self,
        patterns: List[Dict[str, Any]],
        domain_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[Hypothesis]:
        """
        Generate testable hypotheses from patterns
        """
        generated = []
        
        for i, pattern in enumerate(patterns):
            hypothesis = self._pattern_to_hypothesis(pattern, i, domain_knowledge)
            
            if hypothesis and hypothesis.confidence >= self.min_confidence:
                generated.append(hypothesis)
                self.hypotheses.append(hypothesis)
        
        logger.info(f"Generated {len(generated)} hypotheses")
        return generated
    
    def _find_correlations(
        self,
        data: List[Dict[str, Any]],
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Find correlations between variables"""
        correlations = []
        
        numeric_vars = []
        for var in variables:
            try:
                values = [item.get(var) for item in data if var in item]
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    numeric_vars.append(var)
            except:
                pass
        
        # Check pairs
        for i, var1 in enumerate(numeric_vars):
            for var2 in numeric_vars[i+1:]:
                try:
                    vals1 = [item.get(var1) for item in data if var1 in item and var2 in item]
                    vals2 = [item.get(var2) for item in data if var1 in item and var2 in item]
                    
                    if len(vals1) < 3:
                        continue
                    
                    corr = np.corrcoef(vals1, vals2)[0, 1]
                    
                    if abs(corr) > 0.5:  # Significant correlation
                        correlations.append({
                            "type": "correlation",
                            "variable1": var1,
                            "variable2": var2,
                            "coefficient": float(corr),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate",
                            "direction": "positive" if corr > 0 else "negative"
                        })
                except:
                    pass
        
        return correlations
    
    def _find_trends(
        self,
        data: List[Dict[str, Any]],
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Find trends in time-series data"""
        trends = []
        
        # Check if data has temporal ordering
        if "timestamp" in variables or "time" in variables:
            time_var = "timestamp" if "timestamp" in variables else "time"
            
            for var in variables:
                if var == time_var:
                    continue
                
                try:
                    values = [item.get(var) for item in data if var in item]
                    if not all(isinstance(v, (int, float)) for v in values if v is not None):
                        continue
                    
                    # Simple linear trend
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    
                    if abs(slope) > 0.01:  # Detectable trend
                        trends.append({
                            "type": "trend",
                            "variable": var,
                            "slope": float(slope),
                            "direction": "increasing" if slope > 0 else "decreasing",
                            "strength": "strong" if abs(slope) > 0.1 else "weak"
                        })
                except:
                    pass
        
        return trends
    
    def _find_clusters(
        self,
        data: List[Dict[str, Any]],
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Find clusters in data"""
        # Simplified clustering
        clusters = []
        
        # Group by categorical variables
        for var in variables:
            values = [item.get(var) for item in data if var in item]
            
            if len(values) < 5:
                continue
            
            # Check if categorical
            unique_values = list(set(values))
            
            if len(unique_values) < len(values) / 2 and len(unique_values) < 10:
                # Likely categorical
                value_counts = {val: values.count(val) for val in unique_values}
                
                clusters.append({
                    "type": "cluster",
                    "variable": var,
                    "groups": unique_values,
                    "distribution": value_counts
                })
        
        return clusters
    
    def _pattern_to_hypothesis(
        self,
        pattern: Dict[str, Any],
        index: int,
        domain_knowledge: Optional[Dict[str, Any]]
    ) -> Optional[Hypothesis]:
        """Convert discovered pattern to hypothesis"""
        pattern_type = pattern.get("type")
        
        if pattern_type == "correlation":
            statement = (
                f"{pattern['variable1']} and {pattern['variable2']} are "
                f"{pattern['strength']} {pattern['direction']}ly correlated"
            )
            
            variables = [pattern['variable1'], pattern['variable2']]
            confidence = abs(pattern['coefficient'])
            
            predicted_outcomes = {
                "correlation_coefficient": pattern['coefficient'],
                "expected_relationship": pattern['direction']
            }
        
        elif pattern_type == "trend":
            statement = (
                f"{pattern['variable']} shows a {pattern['strength']} "
                f"{pattern['direction']} trend"
            )
            
            variables = [pattern['variable']]
            confidence = min(0.9, abs(pattern['slope']) * 5)
            
            predicted_outcomes = {
                "trend_slope": pattern['slope'],
                "expected_direction": pattern['direction']
            }
        
        elif pattern_type == "cluster":
            statement = (
                f"{pattern['variable']} forms distinct clusters: "
                f"{', '.join(map(str, pattern['groups'][:3]))}"
            )
            
            variables = [pattern['variable']]
            confidence = 0.7
            
            predicted_outcomes = {
                "expected_groups": pattern['groups'],
                "distribution": pattern['distribution']
            }
        
        else:
            return None
        
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{index}_{pattern_type}",
            statement=statement,
            confidence=confidence,
            supporting_evidence=[pattern],
            contradicting_evidence=[],
            testable=True,
            variables=variables,
            predicted_outcomes=predicted_outcomes,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "pattern_type": pattern_type
            }
        )
        
        return hypothesis
    
    def design_experiment(
        self,
        hypothesis: Hypothesis
    ) -> Dict[str, Any]:
        """Design experiment to test hypothesis"""
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "objective": f"Test: {hypothesis.statement}",
            "variables_to_measure": hypothesis.variables,
            "control_variables": [],
            "sample_size_recommendation": 100,
            "measurement_protocol": "Collect data on " + ", ".join(hypothesis.variables),
            "expected_outcomes": hypothesis.predicted_outcomes,
            "success_criteria": f"Confidence > {self.min_confidence}"
        }

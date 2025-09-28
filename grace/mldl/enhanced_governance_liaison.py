"""
Enhanced Governance Liaison for ML/DL Specialist Integration.

Integrates enhanced specialists and cross-domain validators with the governance kernel.
Provides orchestration, consensus building, and governance oversight.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    from .enhanced_specialists import (
        EnhancedMLSpecialist, CrossDomainValidator, SpecialistPrediction, CrossDomainValidation,
        create_enhanced_specialists, create_cross_domain_validators
    )
except ImportError:
    from .specialists.enhanced_specialists import (
        EnhancedMLSpecialist, CrossDomainValidator, SpecialistPrediction, CrossDomainValidation,
        create_enhanced_specialists, create_cross_domain_validators
    )

logger = logging.getLogger(__name__)


class EnhancedGovernanceLiaison:
    """
    Enhanced governance liaison that orchestrates ML/DL specialists and cross-domain validators.
    
    Provides:
    - Enhanced specialist consensus with uncertainty quantification
    - Cross-domain validation and oversight
    - Hallucination detection and mitigation
    - Governance integration with constitutional compliance
    """
    
    def __init__(self, 
                 specialists: Optional[List[EnhancedMLSpecialist]] = None,
                 validators: Optional[List[CrossDomainValidator]] = None,
                 governance_config: Optional[Dict[str, Any]] = None):
        
        self.specialists = specialists or create_enhanced_specialists()
        self.validators = validators or create_cross_domain_validators()
        self.config = governance_config or self._get_default_config()
        
        # Consensus and validation tracking
        self.consensus_history = []
        self.validation_history = []
        
        # Performance metrics
        self.total_requests = 0
        self.successful_consensus = 0
        self.failed_consensus = 0
        self.validation_overrides = 0
        
        self.initialized = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for enhanced governance liaison."""
        return {
            "consensus": {
                "min_specialist_agreement": 0.7,
                "confidence_threshold": 0.6,
                "uncertainty_threshold": 0.4,
                "hallucination_threshold": 0.3
            },
            "validation": {
                "min_validation_score": 0.6,
                "min_compliance_score": 0.7,
                "critical_risk_threshold": 0.8,
                "red_flag_block": True
            },
            "governance": {
                "constitutional_weight": 0.3,
                "specialist_weight": 0.5,
                "validation_weight": 0.2,
                "escalation_threshold": 0.5
            },
            "hallucination_mitigation": {
                "enabled": True,
                "detection_threshold": 0.6,
                "mitigation_strategies": ["evidence_checking", "consensus_verification", "domain_validation"]
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize all specialists and validators."""
        try:
            logger.info("Initializing Enhanced Governance Liaison...")
            
            # Initialize specialists
            specialist_results = []
            for specialist in self.specialists:
                try:
                    result = await specialist.initialize()
                    specialist_results.append(result)
                    if result:
                        logger.info(f"Initialized specialist: {specialist.specialist_id}")
                    else:
                        logger.warning(f"Failed to initialize specialist: {specialist.specialist_id}")
                except Exception as e:
                    logger.error(f"Error initializing specialist {specialist.specialist_id}: {e}")
                    specialist_results.append(False)
            
            # Check if we have enough working specialists
            working_specialists = sum(specialist_results)
            if working_specialists == 0:
                logger.error("No specialists successfully initialized")
                return False
            
            logger.info(f"Initialized {working_specialists}/{len(self.specialists)} specialists")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Governance Liaison: {e}")
            return False
    
    async def process_governance_request(self, 
                                       request_data: Dict[str, Any],
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a governance request with enhanced ML/DL consensus and cross-domain validation.
        """
        if not self.initialized:
            return {"status": "error", "error": "Liaison not initialized"}
        
        try:
            self.total_requests += 1
            request_id = request_data.get("request_id", f"req_{int(datetime.now().timestamp())}")
            
            # Phase 1: Specialist Consensus
            specialist_results = await self._gather_specialist_predictions(request_data, context)
            
            # Phase 2: Build Enhanced Consensus
            consensus_result = await self._build_enhanced_consensus(specialist_results, context)
            
            # Phase 3: Cross-Domain Validation
            validation_results = await self._perform_cross_domain_validation(
                request_data, specialist_results, consensus_result, context
            )
            
            # Phase 4: Hallucination Detection and Mitigation
            hallucination_assessment = await self._assess_hallucination_risk(
                specialist_results, consensus_result, validation_results
            )
            
            # Phase 5: Final Governance Decision
            governance_decision = await self._make_governance_decision(
                request_data, specialist_results, consensus_result, 
                validation_results, hallucination_assessment, context
            )
            
            # Update performance tracking
            if governance_decision["status"] == "success":
                self.successful_consensus += 1
            else:
                self.failed_consensus += 1
            
            # Store in history
            decision_record = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "specialist_results": specialist_results,
                "consensus_result": consensus_result,
                "validation_results": validation_results,
                "hallucination_assessment": hallucination_assessment,
                "governance_decision": governance_decision
            }
            
            self.consensus_history.append(decision_record)
            if len(self.consensus_history) > 1000:
                self.consensus_history = self.consensus_history[-1000:]
            
            return {
                "request_id": request_id,
                "status": "success",
                "specialist_predictions": len(specialist_results),
                "consensus": consensus_result,
                "validation": validation_results["summary"],
                "hallucination_risk": hallucination_assessment["overall_risk"],
                "governance_decision": governance_decision,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced governance processing failed: {e}")
            self.failed_consensus += 1
            return {"status": "error", "error": str(e)}
    
    async def _gather_specialist_predictions(self, 
                                           request_data: Dict[str, Any],
                                           context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gather predictions from all available specialists."""
        predictions = []
        
        for specialist in self.specialists:
            if not specialist.initialized:
                continue
            
            try:
                # Prepare data for this specialist
                specialist_data = self._prepare_specialist_data(request_data, specialist)
                
                # Get prediction with uncertainty
                prediction = await specialist.predict_with_uncertainty(specialist_data, context)
                
                predictions.append({
                    "specialist_id": specialist.specialist_id,
                    "domain": specialist.domain,
                    "prediction": prediction.to_dict(),
                    "specialist_trust": specialist.trust_score
                })
                
            except Exception as e:
                logger.error(f"Specialist {specialist.specialist_id} prediction failed: {e}")
                predictions.append({
                    "specialist_id": specialist.specialist_id,
                    "domain": specialist.domain,
                    "prediction": None,
                    "error": str(e),
                    "specialist_trust": specialist.trust_score
                })
        
        return predictions
    
    def _prepare_specialist_data(self, 
                                request_data: Dict[str, Any], 
                                specialist: EnhancedMLSpecialist) -> Dict[str, Any]:
        """Prepare request data for specific specialist."""
        # Base data
        specialist_data = {
            "task": request_data.get("task_type", "analysis"),
            "data": request_data.get("data", {}),
            "context": request_data.get("context", {})
        }
        
        # Specialist-specific data preparation
        if specialist.domain == "graph_analysis":
            # Extract graph-related data
            if "entities" in request_data or "relationships" in request_data:
                specialist_data.update({
                    "entities": request_data.get("entities", []),
                    "relationships": request_data.get("relationships", []),
                    "dependencies": request_data.get("dependencies", [])
                })
            
        elif specialist.domain == "multimodal_analysis":
            # Extract multimodal data
            if "content" in request_data:
                specialist_data.update({
                    "content": request_data["content"],
                    "content_types": request_data.get("content_types", ["text"]),
                    "modal_content": request_data.get("modal_content", {})
                })
            
        elif specialist.domain == "uncertainty_quantification":
            # Extract prediction data for uncertainty analysis
            specialist_data.update({
                "predictions": request_data.get("predictions", []),
                "model_confidence": request_data.get("model_confidence", []),
                "confidence_scores": request_data.get("confidence_scores", []),
                "actual_accuracies": request_data.get("actual_accuracies", [])
            })
        
        return specialist_data
    
    async def _build_enhanced_consensus(self, 
                                      specialist_results: List[Dict[str, Any]],
                                      context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build enhanced consensus with uncertainty quantification."""
        try:
            valid_predictions = [
                r for r in specialist_results 
                if r.get("prediction") is not None and "error" not in r
            ]
            
            if not valid_predictions:
                return {
                    "consensus_reached": False,
                    "reason": "No valid predictions available",
                    "confidence": 0.0,
                    "uncertainty": 1.0
                }
            
            # Extract prediction metrics
            confidences = []
            uncertainties = []
            hallucination_risks = []
            cross_domain_scores = []
            specialist_trusts = []
            
            for result in valid_predictions:
                pred = result["prediction"]
                confidences.append(pred.get("confidence", 0.0))
                uncertainties.append(pred.get("uncertainty", 1.0))
                hallucination_risks.append(pred.get("hallucination_risk", 0.5))
                cross_domain_scores.append(pred.get("cross_domain_score", 0.5))
                specialist_trusts.append(result.get("specialist_trust", 0.5))
            
            # Calculate weighted consensus metrics
            weights = np.array(specialist_trusts)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            
            consensus_confidence = np.average(confidences, weights=weights)
            consensus_uncertainty = np.average(uncertainties, weights=weights)
            consensus_hallucination_risk = np.average(hallucination_risks, weights=weights)
            consensus_cross_domain_score = np.average(cross_domain_scores, weights=weights)
            
            # Check consensus quality
            confidence_agreement = 1.0 - np.std(confidences)  # High agreement if low std
            uncertainty_agreement = 1.0 - np.std(uncertainties)
            
            overall_agreement = (confidence_agreement + uncertainty_agreement) / 2
            
            # Determine if consensus is reached
            consensus_reached = (
                consensus_confidence >= self.config["consensus"]["confidence_threshold"] and
                consensus_uncertainty <= self.config["consensus"]["uncertainty_threshold"] and
                consensus_hallucination_risk <= self.config["consensus"]["hallucination_threshold"] and
                overall_agreement >= self.config["consensus"]["min_specialist_agreement"]
            )
            
            # Build consensus result
            consensus_result = {
                "consensus_reached": consensus_reached,
                "participating_specialists": len(valid_predictions),
                "consensus_confidence": consensus_confidence,
                "consensus_uncertainty": consensus_uncertainty,
                "consensus_hallucination_risk": consensus_hallucination_risk,
                "consensus_cross_domain_score": consensus_cross_domain_score,
                "specialist_agreement": overall_agreement,
                "individual_predictions": valid_predictions,
                "consensus_weights": weights.tolist()
            }
            
            # Add reasoning
            if consensus_reached:
                consensus_result["reasoning"] = f"Strong consensus reached with {consensus_confidence:.2f} confidence and {overall_agreement:.2f} agreement"
            else:
                reasons = []
                if consensus_confidence < self.config["consensus"]["confidence_threshold"]:
                    reasons.append(f"low confidence ({consensus_confidence:.2f})")
                if consensus_uncertainty > self.config["consensus"]["uncertainty_threshold"]:
                    reasons.append(f"high uncertainty ({consensus_uncertainty:.2f})")
                if consensus_hallucination_risk > self.config["consensus"]["hallucination_threshold"]:
                    reasons.append(f"high hallucination risk ({consensus_hallucination_risk:.2f})")
                if overall_agreement < self.config["consensus"]["min_specialist_agreement"]:
                    reasons.append(f"low agreement ({overall_agreement:.2f})")
                
                consensus_result["reasoning"] = f"Consensus not reached: {', '.join(reasons)}"
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Enhanced consensus building failed: {e}")
            return {
                "consensus_reached": False,
                "reason": f"Consensus building error: {str(e)}",
                "confidence": 0.0,
                "uncertainty": 1.0
            }
    
    async def _perform_cross_domain_validation(self, 
                                             request_data: Dict[str, Any],
                                             specialist_results: List[Dict[str, Any]],
                                             consensus_result: Dict[str, Any],
                                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-domain validation using domain validators."""
        validation_results = {
            "validations": [],
            "summary": {
                "total_validators": len(self.validators),
                "avg_validation_score": 0.0,
                "avg_compliance_score": 0.0,
                "critical_issues": [],
                "red_flags": [],
                "recommendations": []
            }
        }
        
        try:
            # Convert specialist results to SpecialistPrediction objects for validators
            specialist_predictions = []
            for result in specialist_results:
                if result.get("prediction"):
                    pred_data = result["prediction"]
                    prediction = SpecialistPrediction(
                        prediction=pred_data.get("prediction"),
                        confidence=pred_data.get("confidence", 0.0),
                        uncertainty=pred_data.get("uncertainty", 1.0),
                        explanation=pred_data.get("explanation", ""),
                        evidence=pred_data.get("evidence", []),
                        cross_domain_score=pred_data.get("cross_domain_score", 0.0),
                        hallucination_risk=pred_data.get("hallucination_risk", 0.5)
                    )
                    specialist_predictions.append(prediction)
            
            # Run each validator
            validation_scores = []
            compliance_scores = []
            
            for validator in self.validators:
                try:
                    validation = await validator.validate_decision(
                        request_data, specialist_predictions, context
                    )
                    
                    validation_dict = validation.to_dict()
                    validation_results["validations"].append(validation_dict)
                    
                    validation_scores.append(validation.validation_score)
                    compliance_scores.append(validation.compliance_score)
                    
                    # Collect critical issues and red flags
                    if validation.validation_score < self.config["validation"]["min_validation_score"]:
                        validation_results["summary"]["critical_issues"].append(
                            f"{validation.domain} validation score too low: {validation.validation_score:.2f}"
                        )
                    
                    if validation.compliance_score < self.config["validation"]["min_compliance_score"]:
                        validation_results["summary"]["critical_issues"].append(
                            f"{validation.domain} compliance score too low: {validation.compliance_score:.2f}"
                        )
                    
                    validation_results["summary"]["red_flags"].extend(validation.red_flags)
                    validation_results["summary"]["recommendations"].extend(validation.recommendations)
                    
                except Exception as e:
                    logger.error(f"Validation failed for {validator.validator_id}: {e}")
                    validation_results["validations"].append({
                        "validator_id": validator.validator_id,
                        "domain": validator.domain,
                        "error": str(e),
                        "validation_score": 0.0,
                        "compliance_score": 0.0
                    })
            
            # Calculate summary statistics
            if validation_scores:
                validation_results["summary"]["avg_validation_score"] = np.mean(validation_scores)
            if compliance_scores:
                validation_results["summary"]["avg_compliance_score"] = np.mean(compliance_scores)
            
            validation_results["summary"]["validation_passed"] = (
                validation_results["summary"]["avg_validation_score"] >= self.config["validation"]["min_validation_score"] and
                validation_results["summary"]["avg_compliance_score"] >= self.config["validation"]["min_compliance_score"] and
                (not self.config["validation"]["red_flag_block"] or not validation_results["summary"]["red_flags"])
            )
            
        except Exception as e:
            logger.error(f"Cross-domain validation failed: {e}")
            validation_results["summary"]["error"] = str(e)
            validation_results["summary"]["validation_passed"] = False
        
        return validation_results
    
    async def _assess_hallucination_risk(self, 
                                       specialist_results: List[Dict[str, Any]],
                                       consensus_result: Dict[str, Any],
                                       validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess hallucination risk and apply mitigation strategies."""
        try:
            hallucination_assessment = {
                "overall_risk": 0.0,
                "risk_factors": [],
                "mitigation_applied": [],
                "confidence_after_mitigation": 0.0
            }
            
            if not self.config["hallucination_mitigation"]["enabled"]:
                hallucination_assessment["overall_risk"] = consensus_result.get("consensus_hallucination_risk", 0.5)
                return hallucination_assessment
            
            # Analyze risk factors
            risk_factors = []
            
            # Specialist disagreement
            if consensus_result.get("specialist_agreement", 1.0) < 0.7:
                risk_factors.append({
                    "factor": "specialist_disagreement",
                    "risk": 1.0 - consensus_result["specialist_agreement"],
                    "description": "Specialists disagree on predictions"
                })
            
            # High uncertainty
            consensus_uncertainty = consensus_result.get("consensus_uncertainty", 0.0)
            if consensus_uncertainty > 0.4:
                risk_factors.append({
                    "factor": "high_uncertainty",
                    "risk": consensus_uncertainty,
                    "description": f"High prediction uncertainty: {consensus_uncertainty:.2f}"
                })
            
            # Low evidence quality
            valid_predictions = [r for r in specialist_results if r.get("prediction")]
            if valid_predictions:
                evidence_qualities = []
                for result in valid_predictions:
                    evidence = result["prediction"].get("evidence", [])
                    if evidence:
                        avg_quality = np.mean([e.get("quality", 0.5) for e in evidence])
                        evidence_qualities.append(avg_quality)
                
                if evidence_qualities:
                    avg_evidence_quality = np.mean(evidence_qualities)
                    if avg_evidence_quality < 0.6:
                        risk_factors.append({
                            "factor": "low_evidence_quality",
                            "risk": 1.0 - avg_evidence_quality,
                            "description": f"Low evidence quality: {avg_evidence_quality:.2f}"
                        })
            
            # Validation failures
            if not validation_results["summary"].get("validation_passed", True):
                risk_factors.append({
                    "factor": "validation_failure",
                    "risk": 0.8,
                    "description": "Cross-domain validation failed"
                })
            
            # Calculate overall risk
            if risk_factors:
                hallucination_assessment["overall_risk"] = np.mean([rf["risk"] for rf in risk_factors])
                hallucination_assessment["risk_factors"] = risk_factors
            else:
                hallucination_assessment["overall_risk"] = consensus_result.get("consensus_hallucination_risk", 0.0)
            
            # Apply mitigation strategies if risk is high
            mitigation_applied = []
            confidence_adjustment = 1.0
            
            if hallucination_assessment["overall_risk"] > self.config["hallucination_mitigation"]["detection_threshold"]:
                strategies = self.config["hallucination_mitigation"]["mitigation_strategies"]
                
                if "evidence_checking" in strategies:
                    # Downgrade confidence if evidence is weak
                    confidence_adjustment *= 0.8
                    mitigation_applied.append("evidence_checking")
                
                if "consensus_verification" in strategies:
                    # Require higher specialist agreement
                    if consensus_result.get("specialist_agreement", 0.0) < 0.8:
                        confidence_adjustment *= 0.7
                        mitigation_applied.append("consensus_verification")
                
                if "domain_validation" in strategies:
                    # Require validation from all domains
                    failed_validations = [
                        v for v in validation_results["validations"]
                        if v.get("validation_score", 0) < self.config["validation"]["min_validation_score"]
                    ]
                    if failed_validations:
                        confidence_adjustment *= 0.6
                        mitigation_applied.append("domain_validation")
            
            hallucination_assessment["mitigation_applied"] = mitigation_applied
            hallucination_assessment["confidence_after_mitigation"] = (
                consensus_result.get("consensus_confidence", 0.0) * confidence_adjustment
            )
            
            return hallucination_assessment
            
        except Exception as e:
            logger.error(f"Hallucination assessment failed: {e}")
            return {
                "overall_risk": 0.9,
                "error": str(e),
                "mitigation_applied": [],
                "confidence_after_mitigation": 0.0
            }
    
    async def _make_governance_decision(self, 
                                      request_data: Dict[str, Any],
                                      specialist_results: List[Dict[str, Any]],
                                      consensus_result: Dict[str, Any],
                                      validation_results: Dict[str, Any],
                                      hallucination_assessment: Dict[str, Any],
                                      context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make final governance decision integrating all factors."""
        try:
            decision = {
                "status": "success",
                "decision_made": False,
                "decision_type": "reject",
                "confidence": 0.0,
                "reasoning": [],
                "recommendations": [],
                "escalation_required": False
            }
            
            # Extract key metrics
            consensus_confidence = hallucination_assessment.get("confidence_after_mitigation", 0.0)
            consensus_reached = consensus_result.get("consensus_reached", False)
            validation_passed = validation_results["summary"].get("validation_passed", False)
            hallucination_risk = hallucination_assessment.get("overall_risk", 0.5)
            
            # Decision logic
            reasoning = []
            
            # Check for blocking conditions
            red_flags = validation_results["summary"].get("red_flags", [])
            if self.config["validation"]["red_flag_block"] and red_flags:
                decision["decision_type"] = "reject"
                decision["decision_made"] = True
                reasoning.append(f"Blocked by red flags: {', '.join(red_flags)}")
            
            elif hallucination_risk > 0.8:
                decision["decision_type"] = "reject"
                decision["decision_made"] = True
                reasoning.append(f"High hallucination risk: {hallucination_risk:.2f}")
            
            elif not validation_passed:
                if validation_results["summary"]["avg_validation_score"] < 0.3:
                    decision["decision_type"] = "reject"
                    decision["decision_made"] = True
                    reasoning.append("Critical validation failures")
                else:
                    decision["decision_type"] = "conditional_approve"
                    decision["decision_made"] = True
                    reasoning.append("Conditional approval with validation concerns")
            
            elif not consensus_reached:
                if consensus_confidence < 0.3:
                    decision["decision_type"] = "reject"
                    decision["decision_made"] = True
                    reasoning.append("Insufficient specialist consensus")
                else:
                    decision["escalation_required"] = True
                    reasoning.append("Escalation required due to weak consensus")
            
            else:
                # Positive decision logic
                if consensus_confidence > 0.8 and hallucination_risk < 0.2:
                    decision["decision_type"] = "approve"
                    decision["decision_made"] = True
                    reasoning.append("Strong consensus with low risk")
                elif consensus_confidence > 0.6 and validation_passed:
                    decision["decision_type"] = "conditional_approve"
                    decision["decision_made"] = True
                    reasoning.append("Good consensus with validation support")
                else:
                    decision["escalation_required"] = True
                    reasoning.append("Escalation required for human review")
            
            # Calculate final confidence
            governance_weights = self.config["governance"]
            weighted_confidence = (
                consensus_confidence * governance_weights["specialist_weight"] +
                consensus_result.get("consensus_cross_domain_score", 0.5) * governance_weights["constitutional_weight"] +
                validation_results["summary"]["avg_validation_score"] * governance_weights["validation_weight"]
            )
            
            decision["confidence"] = max(0.0, min(1.0, weighted_confidence))
            decision["reasoning"] = reasoning
            
            # Add recommendations
            decision["recommendations"].extend(validation_results["summary"].get("recommendations", []))
            
            if hallucination_assessment.get("mitigation_applied"):
                decision["recommendations"].append(
                    f"Hallucination mitigation applied: {', '.join(hallucination_assessment['mitigation_applied'])}"
                )
            
            # Check escalation threshold
            if decision["confidence"] < self.config["governance"]["escalation_threshold"]:
                decision["escalation_required"] = True
                decision["recommendations"].append("Human oversight recommended due to low confidence")
            
            return decision
            
        except Exception as e:
            logger.error(f"Governance decision making failed: {e}")
            return {
                "status": "error",
                "decision_made": False,
                "decision_type": "error", 
                "confidence": 0.0,
                "reasoning": [f"Decision error: {str(e)}"],
                "escalation_required": True
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the enhanced governance liaison."""
        stats = {
            "initialized": self.initialized,
            "total_requests": self.total_requests,
            "successful_consensus": self.successful_consensus,
            "failed_consensus": self.failed_consensus,
            "success_rate": self.successful_consensus / max(self.total_requests, 1),
            "validation_overrides": self.validation_overrides,
            "specialists": {},
            "validators": {}
        }
        
        # Specialist stats
        for specialist in self.specialists:
            stats["specialists"][specialist.specialist_id] = specialist.get_stats()
        
        # Validator stats
        for validator in self.validators:
            stats["validators"][validator.validator_id] = validator.get_stats()
        
        # Recent performance analysis
        if self.consensus_history:
            recent_decisions = self.consensus_history[-50:]
            
            avg_confidence = np.mean([
                d["consensus_result"].get("consensus_confidence", 0.0) 
                for d in recent_decisions
            ])
            
            avg_hallucination_risk = np.mean([
                d["hallucination_assessment"].get("overall_risk", 0.5) 
                for d in recent_decisions
            ])
            
            escalation_rate = len([
                d for d in recent_decisions 
                if d["governance_decision"].get("escalation_required", False)
            ]) / len(recent_decisions)
            
            stats.update({
                "recent_avg_confidence": avg_confidence,
                "recent_avg_hallucination_risk": avg_hallucination_risk,
                "recent_escalation_rate": escalation_rate
            })
        
        return stats
    
    async def update_specialist_performance(self, 
                                          request_id: str, 
                                          actual_outcome: Any) -> Dict[str, Any]:
        """Update specialist performance based on actual outcomes."""
        try:
            # Find the decision record
            decision_record = None
            for record in self.consensus_history:
                if record["request_id"] == request_id:
                    decision_record = record
                    break
            
            if not decision_record:
                return {"status": "error", "error": "Request not found in history"}
            
            # Update each specialist's performance
            updates = {}
            specialist_results = decision_record["specialist_results"]
            
            for result in specialist_results:
                if result.get("prediction"):
                    specialist_id = result["specialist_id"]
                    prediction = result["prediction"]["prediction"]
                    confidence = result["prediction"]["confidence"]
                    
                    # Find the specialist
                    specialist = next((s for s in self.specialists if s.specialist_id == specialist_id), None)
                    if specialist:
                        specialist.update_performance(actual_outcome, prediction, confidence)
                        updates[specialist_id] = {
                            "old_trust_score": result.get("specialist_trust", 0.0),
                            "new_trust_score": specialist.trust_score
                        }
            
            return {
                "status": "success",
                "request_id": request_id,
                "updates": updates,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced governance liaison."""
        try:
            # No specific shutdown needed for current implementations
            self.initialized = False
            logger.info("Enhanced Governance Liaison shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
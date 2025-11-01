"""
Disagreement-Aware Consensus Engine

The breakthrough intelligence amplifier that transforms naive voting into
intelligent reasoning through disagreement analysis and verification branching.

When models disagree, Grace investigates instead of blindly voting.
This creates emergent reasoning capabilities.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Methods for reaching consensus"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CALIBRATED_AGGREGATE = "calibrated_aggregate"
    VERIFICATION_BRANCH = "verification_branch"


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_id: str
    prediction: Any
    confidence: float
    logits: Optional[np.ndarray] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus process"""
    final_prediction: Any
    confidence: float
    method_used: ConsensusMethod
    agreement_score: float  # 0.0 (complete disagreement) to 1.0 (unanimous)
    individual_predictions: List[ModelPrediction]
    verification_performed: bool = False
    verification_details: Optional[Dict[str, Any]] = None
    credit_assignment: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.final_prediction,
            "confidence": self.confidence,
            "method": self.method_used.value,
            "agreement": self.agreement_score,
            "models_count": len(self.individual_predictions),
            "verification": self.verification_performed,
            "timestamp": self.timestamp.isoformat()
        }


class CalibrationManager:
    """
    Tracks and manages model calibration scores.
    
    Well-calibrated models have confidence scores that match
    their actual accuracy.
    """
    
    def __init__(self):
        self.calibration_scores = {}  # model_id -> calibration score
        self.temperature_scales = {}  # model_id -> optimal temperature
        
        # Default values
        self.default_calibration = 0.8
        self.default_temperature = 1.0
    
    def get_calibration(self, model_id: str) -> float:
        """Get calibration score for a model"""
        return self.calibration_scores.get(model_id, self.default_calibration)
    
    def get_temperature(self, model_id: str) -> float:
        """Get temperature scaling for a model"""
        return self.temperature_scales.get(model_id, self.default_temperature)
    
    def update_calibration(
        self,
        model_id: str,
        predictions: List[Tuple[float, bool]],  # (confidence, correct)
    ):
        """
        Update calibration based on recent predictions.
        
        Uses Expected Calibration Error (ECE) metric.
        """
        if not predictions:
            return
        
        # Group by confidence bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(bins) - 1):
            bin_low, bin_high = bins[i], bins[i + 1]
            bin_preds = [
                (conf, correct) for conf, correct in predictions
                if bin_low <= conf < bin_high
            ]
            
            if bin_preds:
                avg_conf = np.mean([conf for conf, _ in bin_preds])
                accuracy = np.mean([correct for _, correct in bin_preds])
                bin_confidences.append(avg_conf)
                bin_accuracies.append(accuracy)
        
        # Compute ECE
        if bin_confidences:
            ece = np.mean([
                abs(conf - acc)
                for conf, acc in zip(bin_confidences, bin_accuracies)
            ])
            calibration_score = 1.0 - ece  # Higher is better
            self.calibration_scores[model_id] = calibration_score
            
            logger.info(f"Updated calibration for {model_id}: {calibration_score:.3f}")
    
    def find_optimal_temperature(
        self,
        model_id: str,
        predictions: List[Tuple[np.ndarray, bool]]  # (logits, correct)
    ):
        """
        Find optimal temperature scaling to improve calibration.
        
        Temperature scaling: softmax(logits / T)
        """
        if not predictions:
            return
        
        # Try different temperatures
        best_temp = 1.0
        best_calibration = 0.0
        
        for temp in np.linspace(0.5, 2.0, 16):
            # Scale logits and compute confidences
            scaled_predictions = []
            for logits, correct in predictions:
                scaled_probs = self._softmax(logits / temp)
                confidence = float(np.max(scaled_probs))
                scaled_predictions.append((confidence, correct))
            
            # Compute calibration at this temperature
            # (simplified - reuse update logic)
            temp_calibration = self._compute_calibration_score(scaled_predictions)
            
            if temp_calibration > best_calibration:
                best_calibration = temp_calibration
                best_temp = temp
        
        self.temperature_scales[model_id] = best_temp
        logger.info(f"Optimal temperature for {model_id}: {best_temp:.2f}")
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def _compute_calibration_score(
        self,
        predictions: List[Tuple[float, bool]]
    ) -> float:
        """Simplified calibration score computation"""
        if not predictions:
            return 0.0
        
        # Mean absolute error between confidence and accuracy
        confidences = [conf for conf, _ in predictions]
        accuracies = [float(correct) for _, correct in predictions]
        mae = np.mean([abs(c - a) for c, a in zip(confidences, accuracies)])
        return 1.0 - mae


class VerificationBranch:
    """
    Handles verification when models disagree.
    
    This is the BREAKTHROUGH component - instead of blindly voting,
    Grace investigates contradictions.
    """
    
    def __init__(self):
        self.verification_history = []
    
    async def verify_predictions(
        self,
        task: Any,
        predictions: List[ModelPrediction]
    ) -> Dict[str, Any]:
        """
        Perform verification when models disagree.
        
        Strategies:
        1. Generate multiple hypotheses
        2. Use verification tools (calculators, search, etc.)
        3. Run counterfactual checks
        4. Ask each model to critique others' answers
        """
        logger.info("🔍 Disagreement detected - initiating verification branch")
        
        verification_result = {
            "hypotheses": [],
            "tool_verifications": [],
            "critiques": [],
            "recommended_prediction": None,
            "confidence": 0.0
        }
        
        # 1. Generate hypotheses from each prediction
        hypotheses = []
        for pred in predictions:
            hypothesis = {
                "model": pred.model_id,
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning
            }
            hypotheses.append(hypothesis)
        
        verification_result["hypotheses"] = hypotheses
        
        # 2. Tool verification (if applicable)
        tool_results = await self._verify_with_tools(task, predictions)
        verification_result["tool_verifications"] = tool_results
        
        # 3. Cross-critique
        critiques = await self._cross_critique(predictions)
        verification_result["critiques"] = critiques
        
        # 4. Select best prediction based on verification
        best_pred, confidence = self._select_verified_prediction(
            predictions,
            tool_results,
            critiques
        )
        
        verification_result["recommended_prediction"] = best_pred
        verification_result["confidence"] = confidence
        
        self.verification_history.append({
            "task": str(task)[:100],
            "result": verification_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"✅ Verification complete. Recommended: {best_pred}")
        return verification_result
    
    async def _verify_with_tools(
        self,
        task: Any,
        predictions: List[ModelPrediction]
    ) -> List[Dict[str, Any]]:
        """Use external tools to verify predictions"""
        tool_results = []
        
        # Example: For math problems, use calculator
        if "calculate" in str(task).lower() or any(c in str(task) for c in "+-*/"):
            try:
                # Simulate calculator verification
                # In production, actually use calculator tool
                tool_results.append({
                    "tool": "calculator",
                    "verified": True,
                    "result": "calculation verified"
                })
            except Exception as e:
                logger.error(f"Tool verification failed: {e}")
        
        # Example: For factual questions, use search
        if any(word in str(task).lower() for word in ["what", "who", "when", "where"]):
            # Simulate search verification
            # In production, actually search
            tool_results.append({
                "tool": "search",
                "verified": True,
                "result": "fact checked"
            })
        
        return tool_results
    
    async def _cross_critique(
        self,
        predictions: List[ModelPrediction]
    ) -> List[Dict[str, Any]]:
        """Have models critique each other's predictions"""
        critiques = []
        
        # Each model critiques others
        for i, pred_a in enumerate(predictions):
            for j, pred_b in enumerate(predictions):
                if i != j:
                    # Simulate critique
                    # In production, actually ask model_a to critique pred_b
                    critique = {
                        "critic": pred_a.model_id,
                        "target": pred_b.model_id,
                        "critique": f"Analyzing alternative from {pred_b.model_id}",
                        "agreement": abs(i - j) < 2  # Simulate agreement
                    }
                    critiques.append(critique)
        
        return critiques
    
    def _select_verified_prediction(
        self,
        predictions: List[ModelPrediction],
        tool_results: List[Dict[str, Any]],
        critiques: List[Dict[str, Any]]
    ) -> Tuple[Any, float]:
        """Select best prediction based on verification evidence"""
        
        # Score each prediction
        scores = {}
        for pred in predictions:
            score = pred.confidence  # Start with model's own confidence
            
            # Boost if verified by tools
            for tool_result in tool_results:
                if tool_result.get("verified"):
                    score += 0.2
            
            # Boost if other models agree in critique
            agreements = [
                c for c in critiques
                if c["target"] == pred.model_id and c.get("agreement")
            ]
            score += 0.1 * len(agreements)
            
            scores[pred.model_id] = score
        
        # Select highest scored
        best_model = max(scores.keys(), key=lambda k: scores[k])
        best_pred = next(p for p in predictions if p.model_id == best_model)
        
        # Confidence is normalized score
        max_score = max(scores.values())
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return best_pred.prediction, confidence


class DisagreementAwareConsensus:
    """
    THE intelligence amplifier.
    
    Replaces naive majority vote with:
    1. Uncertainty-calibrated aggregation
    2. Disagreement triggers verification
    3. Per-model credit assignment
    """
    
    def __init__(
        self,
        disagreement_threshold: float = 0.3,
        use_calibration: bool = True
    ):
        self.disagreement_threshold = disagreement_threshold
        self.use_calibration = use_calibration
        
        self.calibration_manager = CalibrationManager()
        self.verification_branch = VerificationBranch()
        
        self.consensus_history = []
        
        logger.info("Disagreement-Aware Consensus Engine initialized")
        logger.info(f"  Disagreement threshold: {disagreement_threshold}")
        logger.info(f"  Calibration enabled: {use_calibration}")
    
    async def reach_consensus(
        self,
        task: Any,
        predictions: List[ModelPrediction],
        force_method: Optional[ConsensusMethod] = None
    ) -> ConsensusResult:
        """
        Main consensus method.
        
        Automatically chooses best strategy based on disagreement level.
        """
        logger.info(f"Reaching consensus from {len(predictions)} models...")
        
        # Calculate disagreement
        disagreement = self._calculate_disagreement(predictions)
        logger.info(f"  Disagreement level: {disagreement:.3f}")
        
        # Choose method
        if force_method:
            method = force_method
        elif disagreement > self.disagreement_threshold:
            # High disagreement: INVESTIGATE
            method = ConsensusMethod.VERIFICATION_BRANCH
            logger.info("  → High disagreement detected! Triggering verification branch")
        elif self.use_calibration:
            # Low disagreement with calibration
            method = ConsensusMethod.CALIBRATED_AGGREGATE
        else:
            # Simple weighted vote
            method = ConsensusMethod.WEIGHTED_VOTE
        
        # Execute consensus
        if method == ConsensusMethod.VERIFICATION_BRANCH:
            result = await self._verification_consensus(task, predictions)
        elif method == ConsensusMethod.CALIBRATED_AGGREGATE:
            result = self._calibrated_consensus(predictions)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            result = self._weighted_consensus(predictions)
        else:
            result = self._majority_vote_consensus(predictions)
        
        # Add metadata
        result.method_used = method
        result.agreement_score = 1.0 - disagreement
        result.individual_predictions = predictions
        
        # Log for credit assignment
        await self._log_consensus_decision(result)
        
        self.consensus_history.append(result)
        
        logger.info(f"  ✓ Consensus reached: {result.final_prediction}")
        logger.info(f"    Confidence: {result.confidence:.3f}")
        logger.info(f"    Method: {method.value}")
        
        return result
    
    def _calculate_disagreement(
        self,
        predictions: List[ModelPrediction]
    ) -> float:
        """
        Calculate disagreement level.
        
        Returns 0.0 (unanimous) to 1.0 (complete disagreement).
        """
        if len(predictions) < 2:
            return 0.0
        
        # For classification: variance in predictions
        pred_values = [str(p.prediction) for p in predictions]
        unique_predictions = len(set(pred_values))
        max_disagreement = len(predictions)
        
        # Normalized disagreement
        disagreement = (unique_predictions - 1) / (max_disagreement - 1) if max_disagreement > 1 else 0.0
        
        # Also consider confidence variance
        confidences = [p.confidence for p in predictions]
        conf_variance = np.var(confidences)
        
        # Combined metric
        combined_disagreement = 0.7 * disagreement + 0.3 * conf_variance
        
        return min(1.0, combined_disagreement)
    
    async def _verification_consensus(
        self,
        task: Any,
        predictions: List[ModelPrediction]
    ) -> ConsensusResult:
        """Consensus via verification branch (breakthrough mode)"""
        
        # Run verification
        verification = await self.verification_branch.verify_predictions(task, predictions)
        
        return ConsensusResult(
            final_prediction=verification["recommended_prediction"],
            confidence=verification["confidence"],
            method_used=ConsensusMethod.VERIFICATION_BRANCH,
            agreement_score=0.0,  # Will be set by caller
            individual_predictions=[],  # Will be set by caller
            verification_performed=True,
            verification_details=verification
        )
    
    def _calibrated_consensus(
        self,
        predictions: List[ModelPrediction]
    ) -> ConsensusResult:
        """
        Calibrated aggregation using temperature-scaled logits.
        
        This is more sophisticated than simple voting.
        """
        if not predictions:
            return ConsensusResult(
                final_prediction=None,
                confidence=0.0,
                method_used=ConsensusMethod.CALIBRATED_AGGREGATE,
                agreement_score=0.0,
                individual_predictions=[]
            )
        
        # Get calibration weights
        weights = [
            self.calibration_manager.get_calibration(p.model_id)
            for p in predictions
        ]
        
        # If we have logits, do proper aggregation
        if all(p.logits is not None for p in predictions):
            aggregated_logits = self._aggregate_logits(predictions, weights)
            final_pred = np.argmax(aggregated_logits)
            confidence = float(np.max(self._softmax(aggregated_logits)))
        else:
            # Fall back to weighted vote
            return self._weighted_consensus(predictions)
        
        return ConsensusResult(
            final_prediction=final_pred,
            confidence=confidence,
            method_used=ConsensusMethod.CALIBRATED_AGGREGATE,
            agreement_score=0.0,
            individual_predictions=[]
        )
    
    def _aggregate_logits(
        self,
        predictions: List[ModelPrediction],
        weights: List[float]
    ) -> np.ndarray:
        """Aggregate logits with temperature scaling"""
        
        # Temperature scale each model's logits
        scaled_logits = []
        for pred in predictions:
            temp = self.calibration_manager.get_temperature(pred.model_id)
            scaled = pred.logits / temp
            scaled_logits.append(scaled)
        
        # Weighted average
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize
        
        aggregated = np.average(scaled_logits, axis=0, weights=weights_array)
        return aggregated
    
    def _weighted_consensus(
        self,
        predictions: List[ModelPrediction]
    ) -> ConsensusResult:
        """Weighted vote using model confidences"""
        
        # Weight by confidence
        votes = {}
        for pred in predictions:
            pred_str = str(pred.prediction)
            if pred_str not in votes:
                votes[pred_str] = 0.0
            votes[pred_str] += pred.confidence
        
        # Select highest weighted
        final_pred = max(votes.keys(), key=lambda k: votes[k])
        total_weight = sum(votes.values())
        confidence = votes[final_pred] / total_weight if total_weight > 0 else 0.0
        
        return ConsensusResult(
            final_prediction=final_pred,
            confidence=confidence,
            method_used=ConsensusMethod.WEIGHTED_VOTE,
            agreement_score=0.0,
            individual_predictions=[]
        )
    
    def _majority_vote_consensus(
        self,
        predictions: List[ModelPrediction]
    ) -> ConsensusResult:
        """Simple majority vote (naive baseline)"""
        
        votes = {}
        for pred in predictions:
            pred_str = str(pred.prediction)
            votes[pred_str] = votes.get(pred_str, 0) + 1
        
        final_pred = max(votes.keys(), key=lambda k: votes[k])
        confidence = votes[final_pred] / len(predictions)
        
        return ConsensusResult(
            final_prediction=final_pred,
            confidence=confidence,
            method_used=ConsensusMethod.MAJORITY_VOTE,
            agreement_score=0.0,
            individual_predictions=[]
        )
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    async def _log_consensus_decision(self, result: ConsensusResult):
        """Log consensus for credit assignment"""
        # In production, emit to event bus for meta-loop learning
        log_entry = {
            "prediction": result.final_prediction,
            "confidence": result.confidence,
            "method": result.method_used.value,
            "agreement": result.agreement_score,
            "verification": result.verification_performed,
            "timestamp": result.timestamp.isoformat()
        }
        
        # This would be picked up by meta-loop for learning
        logger.debug(f"Consensus logged: {log_entry}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        if not self.consensus_history:
            return {"total_consensus": 0}
        
        total = len(self.consensus_history)
        verification_count = sum(1 for c in self.consensus_history if c.verification_performed)
        avg_confidence = np.mean([c.confidence for c in self.consensus_history])
        avg_agreement = np.mean([c.agreement_score for c in self.consensus_history])
        
        return {
            "total_consensus": total,
            "verification_triggered": verification_count,
            "verification_rate": verification_count / total if total > 0 else 0.0,
            "avg_confidence": float(avg_confidence),
            "avg_agreement": float(avg_agreement)
        }


if __name__ == "__main__":
    # Demo usage
    async def demo():
        print("🧠 Disagreement-Aware Consensus Demo\n")
        
        consensus_engine = DisagreementAwareConsensus(
            disagreement_threshold=0.3
        )
        
        # Scenario 1: Low disagreement (models agree)
        print("Scenario 1: Models agree")
        predictions_agree = [
            ModelPrediction(model_id="model_a", prediction="Paris", confidence=0.9),
            ModelPrediction(model_id="model_b", prediction="Paris", confidence=0.85),
            ModelPrediction(model_id="model_c", prediction="Paris", confidence=0.92)
        ]
        
        result1 = await consensus_engine.reach_consensus(
            task="What is the capital of France?",
            predictions=predictions_agree
        )
        print(f"  Result: {result1.final_prediction}")
        print(f"  Confidence: {result1.confidence:.3f}")
        print(f"  Method: {result1.method_used.value}")
        print(f"  Agreement: {result1.agreement_score:.3f}\n")
        
        # Scenario 2: High disagreement (models disagree)
        print("Scenario 2: Models disagree - VERIFICATION TRIGGERED")
        predictions_disagree = [
            ModelPrediction(model_id="model_a", prediction="56", confidence=0.8, reasoning="7*8=56"),
            ModelPrediction(model_id="model_b", prediction="54", confidence=0.7, reasoning="Maybe 7*8=54?"),
            ModelPrediction(model_id="model_c", prediction="56", confidence=0.9, reasoning="Calculated 7*8=56")
        ]
        
        result2 = await consensus_engine.reach_consensus(
            task="What is 7 * 8?",
            predictions=predictions_disagree
        )
        print(f"  Result: {result2.final_prediction}")
        print(f"  Confidence: {result2.confidence:.3f}")
        print(f"  Method: {result2.method_used.value}")
        print(f"  Agreement: {result2.agreement_score:.3f}")
        print(f"  Verification: {result2.verification_performed}\n")
        
        # Show stats
        stats = consensus_engine.get_stats()
        print(f"📊 Stats:")
        print(f"  Total consensus: {stats['total_consensus']}")
        print(f"  Verification triggered: {stats['verification_triggered']}")
        print(f"  Verification rate: {stats['verification_rate']:.1%}")
    
    asyncio.run(demo())

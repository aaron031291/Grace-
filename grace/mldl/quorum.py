"""
MLDL Quorum - 21-specialist consensus system for Grace governance kernel.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics
import logging
import json
from enum import Enum
from dataclasses import dataclass, asdict

from ..core.contracts import Experience, generate_correlation_id
from ..core.utils import utc_timestamp

logger = logging.getLogger(__name__)


class SpecialistType(Enum):
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    NLP_TEXT = "nlp_text"
    VISION_CNN = "vision_cnn"
    TIME_SERIES = "time_series_forecasting"
    RECOMMENDERS = "recommenders"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering_segmentation"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction_viz"
    GRAPH_LEARNING = "graph_learning"
    REINFORCEMENT_LEARNING = "rl_agentics"
    BAYESIAN_UNCERTAINTY = "bayesian_uncertainty"
    CAUSAL_INFERENCE = "causal_inference"
    OPTIMIZATION_HPO = "optimization_hpo"
    FAIRNESS_EXPLAINABILITY = "fairness_explainability"
    DATA_QUALITY_DRIFT = "data_quality_drift"
    PRIVACY_SECURITY = "privacy_security"
    AUTOML_PLANNER = "automl_planner"
    EXPERIMENTATION = "experimentation_canary_shadow"
    META_ENSEMBLER = "meta_ensembler"
    GOVERNANCE_LIAISON = "governance_liaison"


@dataclass
class SpecialistOutput:
    """Output from a specialist model."""
    specialist_id: str
    specialist_type: SpecialistType
    prediction: Any
    confidence: float
    probability: Optional[float] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialist_id": self.specialist_id,
            "specialist_type": self.specialist_type.value,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probability": self.probability,
            "metadata": self.metadata or {},
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QuorumConsensus:
    """Result of quorum consensus process."""
    consensus_id: str
    task_type: str
    inputs: Dict[str, Any]
    specialist_outputs: List[SpecialistOutput]
    consensus_prediction: Any
    consensus_confidence: float
    agreement_level: float
    participating_specialists: List[str]
    weighted_vote_breakdown: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_id": self.consensus_id,
            "task_type": self.task_type,
            "inputs": self.inputs,
            "specialist_outputs": [output.to_dict() for output in self.specialist_outputs],
            "consensus_prediction": self.consensus_prediction,
            "consensus_confidence": self.consensus_confidence,
            "agreement_level": self.agreement_level,
            "participating_specialists": self.participating_specialists,
            "weighted_vote_breakdown": self.weighted_vote_breakdown,
            "timestamp": self.timestamp.isoformat()
        }


class SpecialistModel:
    """Base class for specialist models."""
    
    def __init__(self, specialist_id: str, specialist_type: SpecialistType, 
                 confidence_threshold: float = 0.5):
        self.specialist_id = specialist_id
        self.specialist_type = specialist_type
        self.confidence_threshold = confidence_threshold
        self.is_active = True
        self.performance_history = []
        self.domain_expertise = {}
    
    async def predict(self, inputs: Dict[str, Any]) -> SpecialistOutput:
        """Make a prediction based on inputs."""
        start_time = datetime.now()
        
        try:
            # This is a placeholder - real implementations would have actual ML models
            prediction, confidence = await self._generate_prediction(inputs)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            output = SpecialistOutput(
                specialist_id=self.specialist_id,
                specialist_type=self.specialist_type,
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Specialist {self.specialist_id} prediction error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SpecialistOutput(
                specialist_id=self.specialist_id,
                specialist_type=self.specialist_type,
                prediction=None,
                confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def _generate_prediction(self, inputs: Dict[str, Any]) -> Tuple[Any, float]:
        """Generate prediction - to be implemented by specific specialists."""
        # Placeholder implementation
        task_type = inputs.get("task_type", "unknown")
        
        if "classification" in self.specialist_type.value:
            prediction = "approve" if hash(str(inputs)) % 2 == 0 else "reject"
            confidence = 0.7 + (hash(str(inputs)) % 100) / 100 * 0.2
        elif "regression" in self.specialist_type.value:
            prediction = 0.5 + (hash(str(inputs)) % 100 - 50) / 100
            confidence = 0.6 + (hash(str(inputs)) % 100) / 100 * 0.3
        else:
            prediction = {"result": "processed", "score": 0.5}
            confidence = 0.5 + (hash(str(inputs)) % 100) / 100 * 0.4
        
        return prediction, confidence
    
    def update_performance(self, accuracy: float):
        """Update performance metrics."""
        self.performance_history.append({
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_recent_performance(self, window_days: int = 7) -> float:
        """Get recent performance average."""
        if not self.performance_history:
            return 0.5  # Neutral performance
        
        cutoff = datetime.now().timestamp() - (window_days * 24 * 3600)
        recent_scores = [
            entry["accuracy"] for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"]).timestamp() >= cutoff
        ]
        
        return statistics.mean(recent_scores) if recent_scores else 0.5


class EliteNLPSpecialistModel(SpecialistModel):
    """Elite-level NLP Text specialist with advanced language understanding."""
    
    def __init__(self):
        super().__init__("elite_nlp_processor", SpecialistType.NLP_TEXT, confidence_threshold=0.7)
        
        # Initialize elite NLP capabilities
        try:
            from .specialists.elite_nlp_specialist import EliteNLPSpecialist
            self.elite_nlp = EliteNLPSpecialist()
            self.elite_capabilities = True
            logger.info("Elite NLP capabilities loaded for quorum specialist")
        except ImportError:
            self.elite_nlp = None
            self.elite_capabilities = False
            logger.warning("Elite NLP capabilities not available, using basic processing")
    
    async def _generate_prediction(self, inputs: Dict[str, Any]) -> Tuple[Any, float]:
        """Generate elite NLP prediction."""
        
        if "text" not in inputs:
            return {"error": "Text input is required for NLP analysis"}, 0.1
        
        text = inputs["text"]
        task_type = inputs.get("task_type", "analyze")
        context = inputs.get("context", {})
        
        # Use elite NLP if available
        if self.elite_capabilities and self.elite_nlp:
            return await self._elite_nlp_predict(text, task_type, context)
        else:
            return await self._basic_nlp_predict(text, task_type, context)
    
    async def _elite_nlp_predict(self, text: str, task_type: str, context: Dict[str, Any]) -> Tuple[Any, float]:
        """Perform prediction using elite NLP capabilities."""
        
        try:
            if task_type == "sentiment_analysis":
                analysis = await self.elite_nlp.analyze_text(text, context)
                prediction = {
                    "sentiment": analysis.sentiment["label"],
                    "polarity": analysis.sentiment.get("polarity", 0.0),
                    "confidence": analysis.sentiment["confidence"],
                    "subjectivity": analysis.sentiment.get("subjectivity", 0.5)
                }
                return prediction, analysis.sentiment["confidence"]
                
            elif task_type == "entity_extraction":
                analysis = await self.elite_nlp.analyze_text(text, context)
                prediction = {
                    "entities": analysis.entities,
                    "entity_types": list(set(entity["label"] for entity in analysis.entities)),
                    "entity_count": len(analysis.entities)
                }
                return prediction, 0.9
                
            elif task_type == "intent_classification":
                analysis = await self.elite_nlp.analyze_text(text, context)
                prediction = {
                    "intent": analysis.intent["intent"],
                    "intent_type": analysis.intent["intent_type"],
                    "confidence": analysis.intent["confidence"],
                    "urgency": analysis.intent.get("urgency", "normal"),
                    "action_required": analysis.intent.get("action_required", False)
                }
                return prediction, analysis.intent["confidence"]
                
            elif task_type == "toxicity_detection":
                toxicity_result = await self.elite_nlp.detect_toxicity(text)
                prediction = {
                    "is_toxic": toxicity_result["is_toxic"],
                    "toxicity_score": toxicity_result["toxicity_score"],
                    "categories": toxicity_result["categories"],
                    "safe_for_governance": not toxicity_result["is_toxic"]
                }
                return prediction, toxicity_result["confidence"]
                
            elif task_type == "text_summarization":
                analysis = await self.elite_nlp.analyze_text(text, context)
                prediction = {
                    "summary": analysis.summary,
                    "key_topics": [topic["topic"] for topic in analysis.topics[:5]],
                    "readability_score": analysis.readability.get("score", 0.5)
                }
                return prediction, analysis.confidence
                
            else:
                # Comprehensive analysis for governance decisions
                analysis = await self.elite_nlp.analyze_text(text, context)
                
                # Calculate governance suitability score
                governance_score = self._calculate_governance_score(analysis)
                
                prediction = {
                    "comprehensive_analysis": analysis.to_dict(),
                    "governance_recommendation": "approve" if governance_score > 0.7 else "review" if governance_score > 0.4 else "reject",
                    "governance_score": governance_score,
                    "risk_factors": self._identify_risk_factors(analysis),
                    "constitutional_alignment": self._assess_constitutional_alignment(analysis)
                }
                
                return prediction, analysis.confidence
                
        except Exception as e:
            logger.error(f"Elite NLP prediction error: {e}")
            return {"error": str(e), "status": "failed"}, 0.1
    
    async def _basic_nlp_predict(self, text: str, task_type: str, context: Dict[str, Any]) -> Tuple[Any, float]:
        """Fallback basic NLP prediction."""
        
        # Simple keyword-based analysis
        positive_words = ["good", "great", "excellent", "positive", "approve", "accept", "support", "helpful"]
        negative_words = ["bad", "terrible", "awful", "negative", "reject", "deny", "harmful", "toxic"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if task_type == "sentiment_analysis":
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            prediction = {
                "sentiment": sentiment,
                "polarity": (positive_count - negative_count) / max(1, positive_count + negative_count),
                "confidence": confidence
            }
            
            return prediction, confidence
            
        elif task_type == "toxicity_detection":
            toxic_keywords = ["hate", "kill", "stupid", "idiot", "toxic", "harmful"]
            toxic_count = sum(1 for word in toxic_keywords if word in text_lower)
            
            is_toxic = toxic_count > 0
            toxicity_score = min(1.0, toxic_count / 5.0)
            
            prediction = {
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "categories": [word for word in toxic_keywords if word in text_lower],
                "safe_for_governance": not is_toxic
            }
            
            return prediction, 0.7 if toxic_count > 0 else 0.5
            
        else:
            # Basic governance assessment
            governance_score = 0.5
            if positive_count > negative_count:
                governance_score += 0.2
            if any(word in text_lower for word in ["transparency", "fairness", "accountability"]):
                governance_score += 0.2
            if any(word in text_lower for word in ["harm", "bias", "unfair"]):
                governance_score -= 0.3
            
            governance_score = max(0.0, min(1.0, governance_score))
            
            prediction = {
                "governance_recommendation": "approve" if governance_score > 0.7 else "review" if governance_score > 0.4 else "reject",
                "governance_score": governance_score,
                "word_count": len(text.split()),
                "basic_sentiment": "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
            }
            
            return prediction, governance_score
    
    def _calculate_governance_score(self, analysis) -> float:
        """Calculate governance suitability score based on comprehensive analysis."""
        score = 0.5  # Base score
        
        # Sentiment factor
        sentiment = analysis.sentiment.get("label", "neutral")
        if sentiment == "positive":
            score += 0.1
        elif sentiment == "negative":
            score -= 0.1
        
        # Intent factor
        intent_type = analysis.intent.get("intent_type", "informational")
        if intent_type == "transactional" and analysis.intent.get("action_required", False):
            score += 0.15  # Action requests are valuable
        
        # Readability factor
        readability_score = analysis.readability.get("score", 0.5)
        score += (readability_score - 0.5) * 0.2  # Better readability improves score
        
        # Entity factor (more entities suggest richer content)
        entity_count = len(analysis.entities)
        if entity_count > 0:
            score += min(0.15, entity_count * 0.03)
        
        # Linguistic quality factor
        if analysis.linguistic_features.get("lexical_diversity", 0) > 0.7:
            score += 0.1  # Rich vocabulary
        
        return max(0.0, min(1.0, score))
    
    def _identify_risk_factors(self, analysis) -> List[str]:
        """Identify potential risk factors in the content."""
        risks = []
        
        # Sentiment risks
        if analysis.sentiment.get("label") == "negative" and analysis.sentiment.get("confidence", 0) > 0.8:
            risks.append("strong_negative_sentiment")
        
        # Intent risks
        if analysis.intent.get("urgency") == "high":
            risks.append("urgent_request")
        
        # Readability risks
        if analysis.readability.get("grade_level", 8) > 16:
            risks.append("high_complexity")
        
        # Entity risks (too many people mentioned might indicate gossip/politics)
        person_entities = [e for e in analysis.entities if e.get("label") == "PERSON"]
        if len(person_entities) > 5:
            risks.append("multiple_person_references")
        
        return risks
    
    def _assess_constitutional_alignment(self, analysis) -> Dict[str, float]:
        """Assess alignment with constitutional principles."""
        alignment = {
            "transparency": 0.5,
            "fairness": 0.5,
            "accountability": 0.5,
            "consistency": 0.5,
            "harm_prevention": 0.5
        }
        
        # Check for constitutional keywords in the content
        transparency_keywords = ["transparent", "open", "clear", "disclosure", "visible"]
        fairness_keywords = ["fair", "equal", "just", "impartial", "unbiased"]
        accountability_keywords = ["responsible", "accountable", "answerable", "liable"]
        
        # Simple keyword matching for constitutional alignment
        text_content = ' '.join([analysis.text, analysis.summary])
        text_lower = text_content.lower()
        
        if any(keyword in text_lower for keyword in transparency_keywords):
            alignment["transparency"] += 0.3
        if any(keyword in text_lower for keyword in fairness_keywords):
            alignment["fairness"] += 0.3
        if any(keyword in text_lower for keyword in accountability_keywords):
            alignment["accountability"] += 0.3
        
        # Consistency based on linguistic features
        if analysis.linguistic_features.get("lexical_diversity", 0) > 0.6:
            alignment["consistency"] += 0.2
        
        # Harm prevention based on sentiment and toxicity implications
        if analysis.sentiment.get("label") == "positive":
            alignment["harm_prevention"] += 0.2
        
        # Normalize scores
        for key in alignment:
            alignment[key] = max(0.0, min(1.0, alignment[key]))
        
        return alignment


class MLDLQuorum:
    """
    21-specialist quorum system for expert consensus.
    Manages the ensemble of ML/DL specialists and coordinates consensus decisions.
    """
    
    def __init__(self, event_bus, memory_core=None):
        self.event_bus = event_bus
        self.memory_core = memory_core
        self.specialists: Dict[str, SpecialistModel] = {}
        self.specialist_weights: Dict[str, float] = {}
        self.consensus_history: List[QuorumConsensus] = []
        
        # Voting configuration
        self.min_participating_specialists = 11  # Majority of 21
        self.consensus_threshold = 0.65
        self.confidence_threshold = 0.6
        
        # Initialize the 21 specialists
        self._initialize_specialists()
    
    def _initialize_specialists(self):
        """Initialize all 21 specialist models with elite NLP capabilities."""
        specialists_config = [
            # Core ML specialists
            ("tabular_classifier", SpecialistType.TABULAR_CLASSIFICATION, 0.9),
            ("tabular_regressor", SpecialistType.TABULAR_REGRESSION, 0.9),
            ("elite_nlp_processor", SpecialistType.NLP_TEXT, 0.95),  # Enhanced with elite capabilities
            ("vision_analyzer", SpecialistType.VISION_CNN, 0.8),
            ("time_series_forecaster", SpecialistType.TIME_SERIES, 0.85),
            ("recommender_system", SpecialistType.RECOMMENDERS, 0.7),
            ("anomaly_detector", SpecialistType.ANOMALY_DETECTION, 0.9),
            ("clustering_engine", SpecialistType.CLUSTERING, 0.75),
            ("dimensionality_reducer", SpecialistType.DIMENSIONALITY_REDUCTION, 0.7),
            ("graph_analyzer", SpecialistType.GRAPH_LEARNING, 0.8),
            
            # Advanced AI specialists
            ("rl_agent", SpecialistType.REINFORCEMENT_LEARNING, 0.85),
            ("bayesian_modeler", SpecialistType.BAYESIAN_UNCERTAINTY, 0.9),
            ("causal_analyzer", SpecialistType.CAUSAL_INFERENCE, 0.95),
            ("optimizer", SpecialistType.OPTIMIZATION_HPO, 0.8),
            ("fairness_auditor", SpecialistType.FAIRNESS_EXPLAINABILITY, 1.0),
            ("data_quality_monitor", SpecialistType.DATA_QUALITY_DRIFT, 0.85),
            ("security_scanner", SpecialistType.PRIVACY_SECURITY, 1.0),
            ("automl_planner", SpecialistType.AUTOML_PLANNER, 0.8),
            ("experiment_coordinator", SpecialistType.EXPERIMENTATION, 0.75),
            ("meta_ensembler", SpecialistType.META_ENSEMBLER, 0.85),
            ("governance_liaison", SpecialistType.GOVERNANCE_LIAISON, 1.0)
        ]
        
        for specialist_id, specialist_type, weight in specialists_config:
            # Use elite NLP specialist for NLP tasks
            if specialist_type == SpecialistType.NLP_TEXT:
                specialist = EliteNLPSpecialistModel()
            else:
                specialist = SpecialistModel(specialist_id, specialist_type)
            
            self.specialists[specialist_id] = specialist
            self.specialist_weights[specialist_id] = weight
    
    async def request_consensus(self, task_type: str, inputs: Dict[str, Any],
                              required_specialists: Optional[List[str]] = None) -> QuorumConsensus:
        """
        Request consensus from the quorum of specialists.
        
        Args:
            task_type: Type of task for consensus
            inputs: Input data for specialists
            required_specialists: Optional list of required specialists
            
        Returns:
            QuorumConsensus result
        """
        consensus_id = f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generate_correlation_id()}"
        
        try:
            # Select participating specialists
            participating_specialists = await self._select_specialists(
                task_type, inputs, required_specialists
            )
            
            # Get predictions from specialists
            specialist_outputs = await self._collect_specialist_predictions(
                participating_specialists, inputs
            )
            
            # Filter out failed predictions
            valid_outputs = [
                output for output in specialist_outputs 
                if output.prediction is not None and output.confidence >= self.confidence_threshold
            ]
            
            if len(valid_outputs) < self.min_participating_specialists:
                logger.warning(f"Insufficient valid predictions for consensus: {len(valid_outputs)}/{self.min_participating_specialists}")
            
            # Calculate consensus
            consensus = await self._calculate_consensus(
                consensus_id, task_type, inputs, valid_outputs
            )
            
            # Store consensus
            self.consensus_history.append(consensus)
            if len(self.consensus_history) > 10000:  # Keep recent history
                self.consensus_history = self.consensus_history[-10000:]
            
            # Record in memory if available
            if self.memory_core:
                experience = Experience(
                    type="MLDL_CONSENSUS",
                    component_id="mldl_quorum",
                    context={
                        "task_type": task_type,
                        "participating_specialists": len(valid_outputs),
                        "required_specialists": len(required_specialists or [])
                    },
                    outcome={
                        "consensus_confidence": consensus.consensus_confidence,
                        "agreement_level": consensus.agreement_level
                    },
                    success_score=consensus.consensus_confidence,
                    timestamp=datetime.now()
                )
                self.memory_core.store_experience(experience)
            
            # Publish consensus event
            await self.event_bus.publish("MLDL_CONSENSUS_REACHED", consensus.to_dict())
            
            logger.info(f"Consensus {consensus_id} reached with {len(valid_outputs)} specialists (confidence: {consensus.consensus_confidence:.3f})")
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error in consensus request: {e}")
            return self._create_error_consensus(consensus_id, task_type, inputs, str(e))
    
    async def _select_specialists(self, task_type: str, inputs: Dict[str, Any],
                                required_specialists: Optional[List[str]] = None) -> List[str]:
        """Select which specialists should participate in consensus."""
        selected = set()
        
        # Always include required specialists
        if required_specialists:
            for specialist_id in required_specialists:
                if specialist_id in self.specialists and self.specialists[specialist_id].is_active:
                    selected.add(specialist_id)
        
        # Always include governance liaison
        if "governance_liaison" in self.specialists:
            selected.add("governance_liaison")
        
        # Add specialists based on task type relevance
        task_relevance = {
            "classification": [
                "tabular_classifier", "nlp_processor", "vision_analyzer",
                "fairness_auditor", "meta_ensembler"
            ],
            "regression": [
                "tabular_regressor", "time_series_forecaster", "bayesian_modeler",
                "causal_analyzer", "optimizer"
            ],
            "anomaly_detection": [
                "anomaly_detector", "security_scanner", "data_quality_monitor",
                "clustering_engine", "bayesian_modeler"
            ],
            "recommendation": [
                "recommender_system", "clustering_engine", "graph_analyzer",
                "fairness_auditor", "privacy_security"
            ],
            "governance": [
                "fairness_auditor", "security_scanner", "governance_liaison",
                "causal_analyzer", "bayesian_modeler", "experiment_coordinator"
            ],
            "general": [
                "tabular_classifier", "anomaly_detector", "fairness_auditor",
                "security_scanner", "governance_liaison", "meta_ensembler"
            ]
        }
        
        relevant_specialists = task_relevance.get(task_type, task_relevance["general"])
        for specialist_id in relevant_specialists:
            if specialist_id in self.specialists and self.specialists[specialist_id].is_active:
                selected.add(specialist_id)
        
        # Ensure minimum participation by adding highest weighted specialists
        if len(selected) < self.min_participating_specialists:
            remaining_specialists = [
                (specialist_id, self.specialist_weights.get(specialist_id, 0.5))
                for specialist_id in self.specialists.keys()
                if (specialist_id not in selected and 
                    self.specialists[specialist_id].is_active)
            ]
            
            # Sort by weight and add highest weighted
            remaining_specialists.sort(key=lambda x: x[1], reverse=True)
            needed = self.min_participating_specialists - len(selected)
            
            for specialist_id, _ in remaining_specialists[:needed]:
                selected.add(specialist_id)
        
        return list(selected)
    
    async def _collect_specialist_predictions(self, specialist_ids: List[str],
                                            inputs: Dict[str, Any]) -> List[SpecialistOutput]:
        """Collect predictions from selected specialists."""
        prediction_tasks = []
        
        for specialist_id in specialist_ids:
            if specialist_id in self.specialists:
                specialist = self.specialists[specialist_id]
                task = specialist.predict(inputs)
                prediction_tasks.append(task)
        
        # Run predictions concurrently with timeout
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*prediction_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout
            )
            
            # Filter out exceptions
            valid_outputs = [
                output for output in outputs 
                if isinstance(output, SpecialistOutput)
            ]
            
            return valid_outputs
            
        except asyncio.TimeoutError:
            logger.warning("Specialist prediction timeout")
            return []
    
    async def _calculate_consensus(self, consensus_id: str, task_type: str,
                                 inputs: Dict[str, Any], 
                                 outputs: List[SpecialistOutput]) -> QuorumConsensus:
        """Calculate consensus from specialist outputs."""
        if not outputs:
            return self._create_error_consensus(consensus_id, task_type, inputs, "No valid predictions")
        
        # Calculate weighted votes
        weighted_votes = {}
        total_weight = 0
        confidence_scores = []
        
        for output in outputs:
            weight = self.specialist_weights.get(output.specialist_id, 0.5)
            # Adjust weight by confidence
            adjusted_weight = weight * output.confidence
            
            prediction_key = self._normalize_prediction(output.prediction)
            weighted_votes[prediction_key] = weighted_votes.get(prediction_key, 0) + adjusted_weight
            total_weight += adjusted_weight
            confidence_scores.append(output.confidence)
        
        # Normalize weighted votes
        if total_weight > 0:
            for key in weighted_votes:
                weighted_votes[key] = weighted_votes[key] / total_weight
        
        # Determine consensus prediction
        if weighted_votes:
            consensus_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]
            consensus_weight = weighted_votes[consensus_prediction]
        else:
            consensus_prediction = "no_consensus"
            consensus_weight = 0.0
        
        # Calculate confidence and agreement
        consensus_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
        
        # Agreement level based on vote distribution
        if len(weighted_votes) <= 1:
            agreement_level = 1.0
        else:
            sorted_votes = sorted(weighted_votes.values(), reverse=True)
            top_vote = sorted_votes[0]
            second_vote = sorted_votes[1] if len(sorted_votes) > 1 else 0
            agreement_level = (top_vote - second_vote) / top_vote if top_vote > 0 else 0.0
        
        # Adjust consensus confidence based on agreement
        consensus_confidence = consensus_confidence * (0.5 + 0.5 * agreement_level)
        
        return QuorumConsensus(
            consensus_id=consensus_id,
            task_type=task_type,
            inputs=inputs,
            specialist_outputs=outputs,
            consensus_prediction=self._denormalize_prediction(consensus_prediction),
            consensus_confidence=consensus_confidence,
            agreement_level=agreement_level,
            participating_specialists=[output.specialist_id for output in outputs],
            weighted_vote_breakdown=weighted_votes,
            timestamp=datetime.now()
        )
    
    def _normalize_prediction(self, prediction: Any) -> str:
        """Normalize prediction to a comparable key."""
        if prediction is None:
            return "null"
        elif isinstance(prediction, bool):
            return str(prediction).lower()
        elif isinstance(prediction, (int, float)):
            # Bin numeric predictions for voting
            if prediction < 0.33:
                return "low"
            elif prediction < 0.67:
                return "medium"
            else:
                return "high"
        elif isinstance(prediction, str):
            return prediction.lower().strip()
        elif isinstance(prediction, dict):
            # Use a key from the dict or hash
            if "class" in prediction:
                return str(prediction["class"]).lower()
            elif "result" in prediction:
                return str(prediction["result"]).lower()
            else:
                return str(hash(json.dumps(prediction, sort_keys=True)))
        else:
            return str(prediction).lower()
    
    def _denormalize_prediction(self, prediction_key: str) -> Any:
        """Convert normalized prediction key back to a meaningful prediction."""
        if prediction_key == "null":
            return None
        elif prediction_key in ["true", "false"]:
            return prediction_key == "true"
        elif prediction_key in ["low", "medium", "high"]:
            return {"category": prediction_key, "confidence_band": prediction_key}
        else:
            return prediction_key
    
    def _create_error_consensus(self, consensus_id: str, task_type: str,
                              inputs: Dict[str, Any], error_msg: str) -> QuorumConsensus:
        """Create an error consensus when consensus fails."""
        return QuorumConsensus(
            consensus_id=consensus_id,
            task_type=task_type,
            inputs=inputs,
            specialist_outputs=[],
            consensus_prediction=None,
            consensus_confidence=0.0,
            agreement_level=0.0,
            participating_specialists=[],
            weighted_vote_breakdown={},
            timestamp=datetime.now()
        )
    
    def update_specialist_weight(self, specialist_id: str, new_weight: float):
        """Update the weight of a specialist based on performance."""
        if specialist_id in self.specialist_weights:
            old_weight = self.specialist_weights[specialist_id]
            self.specialist_weights[specialist_id] = max(0.1, min(1.5, new_weight))
            logger.info(f"Updated specialist {specialist_id} weight: {old_weight:.3f} -> {new_weight:.3f}")
    
    def deactivate_specialist(self, specialist_id: str, reason: str = ""):
        """Deactivate a specialist."""
        if specialist_id in self.specialists:
            self.specialists[specialist_id].is_active = False
            logger.info(f"Deactivated specialist {specialist_id}: {reason}")
    
    def activate_specialist(self, specialist_id: str):
        """Activate a specialist."""
        if specialist_id in self.specialists:
            self.specialists[specialist_id].is_active = True
            logger.info(f"Activated specialist {specialist_id}")
    
    def get_quorum_status(self) -> Dict[str, Any]:
        """Get current status of the quorum."""
        active_specialists = sum(1 for s in self.specialists.values() if s.is_active)
        total_specialists = len(self.specialists)
        
        recent_consensus = self.consensus_history[-10:] if self.consensus_history else []
        avg_confidence = statistics.mean([c.consensus_confidence for c in recent_consensus]) if recent_consensus else 0.0
        avg_agreement = statistics.mean([c.agreement_level for c in recent_consensus]) if recent_consensus else 0.0
        
        return {
            "total_specialists": total_specialists,
            "active_specialists": active_specialists,
            "consensus_count": len(self.consensus_history),
            "recent_avg_confidence": avg_confidence,
            "recent_avg_agreement": avg_agreement,
            "specialist_weights": self.specialist_weights.copy(),
            "consensus_threshold": self.consensus_threshold,
            "min_participating": self.min_participating_specialists
        }
    
    def get_specialist_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all specialists."""
        performance = {}
        
        for specialist_id, specialist in self.specialists.items():
            recent_performance = specialist.get_recent_performance()
            weight = self.specialist_weights.get(specialist_id, 0.5)
            
            performance[specialist_id] = {
                "specialist_type": specialist.specialist_type.value,
                "is_active": specialist.is_active,
                "weight": weight,
                "recent_performance": recent_performance,
                "total_predictions": len(specialist.performance_history)
            }
        
        return performance
    
    async def recalibrate_weights(self):
        """Recalibrate specialist weights based on recent performance."""
        for specialist_id, specialist in self.specialists.items():
            recent_performance = specialist.get_recent_performance()
            current_weight = self.specialist_weights.get(specialist_id, 0.5)
            
            # Adjust weight based on performance (simple approach)
            if recent_performance > 0.8:
                new_weight = min(1.5, current_weight * 1.05)
            elif recent_performance < 0.4:
                new_weight = max(0.1, current_weight * 0.95)
            else:
                new_weight = current_weight  # No change
            
            if abs(new_weight - current_weight) > 0.01:
                self.update_specialist_weight(specialist_id, new_weight)
"""Elite NLP Specialist - Advanced natural language processing for Grace governance system."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import re

# Advanced NLP dependencies
try:
    import spacy
    import nltk
    from textblob import TextBlob
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForTokenClassification,
        BertTokenizer,
        BertModel,
    )
    import torch

    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

# Basic NLP fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NLPAnalysis:
    """Comprehensive NLP analysis result."""

    text: str
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    intent: Dict[str, Any]
    topics: List[Dict[str, float]]
    readability: Dict[str, float]
    linguistic_features: Dict[str, Any]
    embeddings: Optional[List[float]]
    summary: str
    confidence: float
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_length": len(self.text),
            "sentiment": self.sentiment,
            "entities": self.entities,
            "intent": self.intent,
            "topics": self.topics,
            "readability": self.readability,
            "linguistic_features": self.linguistic_features,
            "embeddings_dim": len(self.embeddings) if self.embeddings else 0,
            "summary": self.summary,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "analyzed_at": datetime.now().isoformat(),
        }


@dataclass
class ConversationContext:
    """Context for conversational understanding."""

    conversation_id: str
    turn_number: int
    previous_turns: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    domain: Optional[str]
    session_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "previous_turns_count": len(self.previous_turns),
            "user_profile": self.user_profile,
            "domain": self.domain,
            "session_state": self.session_state,
        }


class EliteNLPSpecialist:
    """Elite-level NLP specialist with advanced language understanding capabilities."""

    def __init__(self):
        self.specialist_id = "elite_nlp_specialist"
        self.version = "1.0.0-elite"
        self.capabilities = [
            "sentiment_analysis",
            "named_entity_recognition",
            "intent_classification",
            "text_summarization",
            "question_answering",
            "topic_modeling",
            "readability_analysis",
            "linguistic_analysis",
            "conversation_management",
            "multilingual_support",
            "toxicity_detection",
            "bias_detection",
        ]

        self.models = {}
        self.conversation_contexts = {}
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
        }

        self._initialize_models()
        logger.info(f"Elite NLP Specialist {self.specialist_id} initialized")

    def _initialize_models(self):
        """Initialize all NLP models and components."""
        if not ADVANCED_NLP_AVAILABLE:
            logger.warning(
                "Advanced NLP libraries not available. Using basic fallbacks."
            )
            self._initialize_basic_models()
            return

        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")

            # Initialize sentiment analysis
            self.models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1,
            )

            # Initialize named entity recognition
            self.models["ner"] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
            )

            # Initialize question answering
            self.models["qa"] = pipeline(
                "question-answering", model="deepset/roberta-base-squad2"
            )

            # Initialize text summarization
            self.models["summarization"] = pipeline(
                "summarization", model="facebook/bart-large-cnn"
            )

            # Initialize toxicity detection
            self.models["toxicity"] = pipeline(
                "text-classification", model="unitary/toxic-bert"
            )

            # Initialize intent classification
            self.models["intent"] = pipeline(
                "text-classification", model="microsoft/DialoGPT-medium"
            )

            # Initialize embeddings model
            self.models["embeddings"] = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            logger.info("Advanced NLP models initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing advanced models: {e}")
            self._initialize_basic_models()

    def _initialize_basic_models(self):
        """Initialize basic fallback models."""
        self.nlp = None
        self.models = {}

        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english"
            )
            logger.info("Basic NLP models initialized with sklearn")
        else:
            self.tfidf_vectorizer = None
            logger.warning("No NLP libraries available. Using regex-based processing.")

    async def analyze_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> NLPAnalysis:
        """Perform comprehensive NLP analysis on text."""
        start_time = datetime.now()

        try:
            # Initialize analysis components
            sentiment = await self._analyze_sentiment(text)
            entities = await self._extract_entities(text)
            intent = await self._classify_intent(text, context)
            topics = await self._extract_topics(text)
            readability = await self._analyze_readability(text)
            linguistic_features = await self._extract_linguistic_features(text)
            embeddings = await self._get_embeddings(text)
            summary = await self._summarize_text(text)

            # Calculate confidence based on model certainties
            confidence = self._calculate_confidence(
                [
                    sentiment.get("confidence", 0.5),
                    intent.get("confidence", 0.5),
                    readability.get(
                        "confidence", 0.8
                    ),  # Readability is generally reliable
                ]
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            analysis = NLPAnalysis(
                text=text,
                sentiment=sentiment,
                entities=entities,
                intent=intent,
                topics=topics,
                readability=readability,
                linguistic_features=linguistic_features,
                embeddings=embeddings,
                summary=summary,
                confidence=confidence,
                processing_time=processing_time,
            )

            # Update performance metrics
            self._update_metrics(True, confidence, processing_time)

            return analysis

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, 0.0, processing_time)

            # Return basic analysis
            return await self._basic_text_analysis(text, processing_time)

    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with advanced models or fallback."""
        try:
            if "sentiment" in self.models:
                result = self.models["sentiment"](text[:500])  # Limit length
                return {
                    "label": result[0]["label"].lower(),
                    "score": float(result[0]["score"]),
                    "confidence": float(result[0]["score"]),
                    "polarity": self._convert_to_polarity(
                        result[0]["label"], result[0]["score"]
                    ),
                }
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                return {
                    "label": "positive"
                    if blob.sentiment.polarity > 0
                    else "negative"
                    if blob.sentiment.polarity < 0
                    else "neutral",
                    "score": abs(blob.sentiment.polarity),
                    "confidence": 0.7,
                    "polarity": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                }
        except Exception as e:
            logger.warning(f"Sentiment analysis error: {e}")
            return {
                "label": "neutral",
                "score": 0.5,
                "confidence": 0.1,
                "polarity": 0.0,
            }

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using advanced NER or fallback."""
        entities = []

        try:
            # Advanced NER
            if "ner" in self.models:
                ner_results = self.models["ner"](text)
                for entity in ner_results:
                    entities.append(
                        {
                            "text": entity["word"],
                            "label": entity["entity_group"],
                            "confidence": float(entity["score"]),
                            "start": int(entity.get("start", 0)),
                            "end": int(entity.get("end", 0)),
                        }
                    )

            # spaCy NER
            elif self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "confidence": 0.8,
                            "start": ent.start_char,
                            "end": ent.end_char,
                        }
                    )

            # Fallback regex-based entity extraction
            else:
                entities = await self._extract_entities_regex(text)

        except Exception as e:
            logger.warning(f"Entity extraction error: {e}")
            entities = await self._extract_entities_regex(text)

        return entities

    async def _extract_entities_regex(self, text: str) -> List[Dict[str, Any]]:
        """Fallback regex-based entity extraction."""
        entities = []

        # Person names
        person_pattern = r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"
        for match in re.finditer(person_pattern, text):
            entities.append(
                {
                    "text": match.group(1),
                    "label": "PERSON",
                    "confidence": 0.6,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        # Dates
        date_pattern = r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b"
        for match in re.finditer(date_pattern, text):
            entities.append(
                {
                    "text": match.group(1),
                    "label": "DATE",
                    "confidence": 0.8,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:\([^\s<>"{}|\\^`\[\]]*\))?'
        for match in re.finditer(url_pattern, text):
            entities.append(
                {
                    "text": match.group(0),
                    "label": "URL",
                    "confidence": 0.9,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        return entities

    async def _classify_intent(
        self, text: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify user intent with contextual understanding."""
        intent_classification = {
            "intent": "unknown",
            "confidence": 0.0,
            "intent_type": "informational",
            "domain": "general",
            "urgency": "normal",
            "action_required": False,
        }

        try:
            # Question detection
            question_patterns = [
                r"\b(what|how|when|where|why|who|which|can|could|would|should|do|does|did|is|are|will)\b",
                r"\?$",
            ]

            text_lower = text.lower()
            is_question = any(
                re.search(pattern, text_lower) for pattern in question_patterns
            )

            # Action detection
            action_patterns = [
                r"\b(create|make|build|generate|produce)\b",
                r"\b(update|modify|change|edit|alter|fix)\b",
                r"\b(delete|remove|destroy|cancel)\b",
                r"\b(find|search|look|locate|discover)\b",
                r"\b(analyze|examine|review|check|test)\b",
                r"\b(help|assist|support|guide)\b",
            ]

            has_action = any(
                re.search(pattern, text_lower) for pattern in action_patterns
            )

            # Urgency detection
            urgent_keywords = [
                "urgent",
                "asap",
                "immediately",
                "critical",
                "emergency",
                "now",
            ]
            is_urgent = any(keyword in text_lower for keyword in urgent_keywords)

            # Classify intent
            if is_question:
                intent_classification["intent"] = "question"
                intent_classification["intent_type"] = "informational"
                intent_classification["confidence"] = 0.8
            elif has_action:
                intent_classification["intent"] = "action_request"
                intent_classification["intent_type"] = "transactional"
                intent_classification["action_required"] = True
                intent_classification["confidence"] = 0.7
            else:
                intent_classification["intent"] = "statement"
                intent_classification["intent_type"] = "informational"
                intent_classification["confidence"] = 0.6

            # Set urgency
            if is_urgent:
                intent_classification["urgency"] = "high"

            # Context-based refinement
            if context:
                if "domain" in context:
                    intent_classification["domain"] = context["domain"]

                if "user_history" in context:
                    # Use user history to improve classification
                    intent_classification["confidence"] = min(
                        1.0, intent_classification["confidence"] + 0.1
                    )

        except Exception as e:
            logger.warning(f"Intent classification error: {e}")

        return intent_classification

    async def _extract_topics(self, text: str) -> List[Dict[str, float]]:
        """Extract topics using advanced topic modeling or keyword extraction."""
        topics = []

        try:
            if self.nlp:
                doc = self.nlp(text)

                # Extract noun phrases as potential topics
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

                # Count frequency and calculate relevance
                phrase_counts = {}
                for phrase in noun_phrases:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

                # Convert to topic format
                total_phrases = len(noun_phrases)
                for phrase, count in sorted(
                    phrase_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    topics.append(
                        {
                            "topic": phrase,
                            "score": count / total_phrases,
                            "relevance": min(1.0, count / 3.0),  # Normalize relevance
                        }
                    )

            else:
                # Fallback keyword extraction
                keywords = self._extract_keywords_simple(text)
                for i, keyword in enumerate(keywords[:10]):
                    topics.append(
                        {
                            "topic": keyword,
                            "score": (10 - i) / 10.0,  # Decreasing score
                            "relevance": 0.7,
                        }
                    )

        except Exception as e:
            logger.warning(f"Topic extraction error: {e}")

        return topics

    async def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze text readability using multiple metrics."""
        try:
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not words or not sentences:
                return {"score": 0.0, "grade_level": 0.0, "confidence": 0.1}

            # Basic metrics
            avg_words_per_sentence = len(words) / len(sentences)
            avg_chars_per_word = sum(len(word) for word in words) / len(words)

            # Simple readability score (Flesch-like)
            readability_score = max(
                0,
                min(
                    100,
                    206.835
                    - (1.015 * avg_words_per_sentence)
                    - (84.6 * avg_chars_per_word / 5),
                ),
            )

            # Estimated grade level
            grade_level = max(
                1,
                min(
                    16,
                    0.39 * avg_words_per_sentence
                    + 11.8 * avg_chars_per_word / 5
                    - 15.59,
                ),
            )

            return {
                "score": readability_score / 100.0,  # Normalize to 0-1
                "grade_level": grade_level,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_chars_per_word": avg_chars_per_word,
                "confidence": 0.8,
            }

        except Exception as e:
            logger.warning(f"Readability analysis error: {e}")
            return {"score": 0.5, "grade_level": 8.0, "confidence": 0.1}

    async def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract advanced linguistic features."""
        features = {
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "punctuation_density": 0.0,
            "capitalization_ratio": 0.0,
            "question_count": 0,
            "exclamation_count": 0,
            "avg_sentence_length": 0.0,
            "lexical_diversity": 0.0,
        }

        try:
            # Basic counts
            words = text.split()
            sentences = re.split(r"[.!?]+", text)
            paragraphs = text.split("\n\n")

            features["word_count"] = len(words)
            features["sentence_count"] = len([s for s in sentences if s.strip()])
            features["paragraph_count"] = len([p for p in paragraphs if p.strip()])

            if features["word_count"] > 0:
                # Punctuation density
                punctuation_chars = len(re.findall(r"[.!?,:;]", text))
                features["punctuation_density"] = (
                    punctuation_chars / features["word_count"]
                )

                # Capitalization ratio
                capital_chars = len(re.findall(r"[A-Z]", text))
                features["capitalization_ratio"] = (
                    capital_chars / len(text.replace(" ", ""))
                    if len(text.replace(" ", "")) > 0
                    else 0
                )

                # Question and exclamation counts
                features["question_count"] = text.count("?")
                features["exclamation_count"] = text.count("!")

                # Average sentence length
                if features["sentence_count"] > 0:
                    features["avg_sentence_length"] = (
                        features["word_count"] / features["sentence_count"]
                    )

                # Lexical diversity (type-token ratio)
                unique_words = set(word.lower() for word in words)
                features["lexical_diversity"] = (
                    len(unique_words) / features["word_count"]
                )

            # Advanced features with spaCy
            if self.nlp:
                doc = self.nlp(text)

                # POS tag distribution
                pos_counts = {}
                for token in doc:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

                features["pos_distribution"] = pos_counts

                # Dependency analysis
                dep_counts = {}
                for token in doc:
                    dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1

                features["dependency_distribution"] = dep_counts

        except Exception as e:
            logger.warning(f"Linguistic feature extraction error: {e}")

        return features

    async def _get_embeddings(self, text: str) -> Optional[List[float]]:
        """Get text embeddings using advanced models or fallback."""
        try:
            if "embeddings" in self.models and self.tokenizer:
                # Use BERT embeddings
                inputs = self.tokenizer(
                    text[:512], return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    outputs = self.models["embeddings"](**inputs)
                    embeddings = (
                        outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                    )
                return embeddings

            elif self.tfidf_vectorizer:
                # Use TF-IDF as fallback
                try:
                    # Fit if not already fitted
                    self.tfidf_vectorizer.fit([text])
                    tfidf_matrix = self.tfidf_vectorizer.transform([text])
                    return tfidf_matrix.toarray()[0].tolist()
                except Exception:
                    # If fitting fails, return None
                    return None

        except Exception as e:
            logger.warning(f"Embedding extraction error: {e}")

        return None

    async def _summarize_text(self, text: str) -> str:
        """Generate text summary using advanced models or extractive approach."""
        try:
            if "summarization" in self.models and len(text) > 100:
                # Use transformer summarization
                summary_result = self.models["summarization"](
                    text[:1000]
                )  # Limit input length
                return summary_result[0]["summary_text"]

            else:
                # Fallback: extractive summarization (first sentence + keywords)
                sentences = re.split(r"[.!?]+", text)
                if sentences:
                    first_sentence = sentences[0].strip()
                    keywords = self._extract_keywords_simple(text)[:5]
                    keyword_str = ", ".join(keywords) if keywords else ""
                    return (
                        f"{first_sentence}. Key topics: {keyword_str}"
                        if keyword_str
                        else first_sentence
                    )

        except Exception as e:
            logger.warning(f"Text summarization error: {e}")

        # Ultimate fallback
        return text[:100] + "..." if len(text) > 100 else text

    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction using frequency analysis."""
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words = [word for word in words if word not in stop_words]

        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Return top keywords
        return [
            word
            for word, count in sorted(
                word_freq.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _convert_to_polarity(self, label: str, score: float) -> float:
        """Convert sentiment label and score to polarity (-1 to 1)."""
        label_lower = label.lower()
        if "positive" in label_lower:
            return score
        elif "negative" in label_lower:
            return -score
        else:
            return 0.0

    def _calculate_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence from individual model confidences."""
        if not confidences:
            return 0.0

        # Use weighted average with emphasis on higher confidences
        weights = [conf**2 for conf in confidences]  # Square to emphasize higher values
        if sum(weights) == 0:
            return 0.0

        weighted_avg = sum(
            conf * weight for conf, weight in zip(confidences, weights)
        ) / sum(weights)
        return min(1.0, weighted_avg)

    async def _basic_text_analysis(
        self, text: str, processing_time: float
    ) -> NLPAnalysis:
        """Fallback basic text analysis when advanced models fail."""
        return NLPAnalysis(
            text=text,
            sentiment={
                "label": "neutral",
                "score": 0.5,
                "confidence": 0.1,
                "polarity": 0.0,
            },
            entities=[],
            intent={
                "intent": "unknown",
                "confidence": 0.1,
                "intent_type": "informational",
            },
            topics=[],
            readability={"score": 0.5, "grade_level": 8.0, "confidence": 0.1},
            linguistic_features={"word_count": len(text.split())},
            embeddings=None,
            summary=text[:100] + "..." if len(text) > 100 else text,
            confidence=0.1,
            processing_time=processing_time,
        )

    def _update_metrics(self, success: bool, confidence: float, processing_time: float):
        """Update performance metrics."""
        self.performance_metrics["total_requests"] += 1

        if success:
            self.performance_metrics["successful_requests"] += 1

        # Update running averages
        total = self.performance_metrics["total_requests"]
        self.performance_metrics["average_confidence"] = (
            self.performance_metrics["average_confidence"] * (total - 1) + confidence
        ) / total
        self.performance_metrics["average_processing_time"] = (
            self.performance_metrics["average_processing_time"] * (total - 1)
            + processing_time
        ) / total

    async def manage_conversation_context(
        self,
        conversation_id: str,
        user_input: str,
        user_profile: Dict[str, Any],
        domain: Optional[str] = None,
    ) -> ConversationContext:
        """Manage conversation context for improved understanding."""
        # Get or create conversation context
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                "turns": [],
                "session_state": {},
                "created_at": datetime.now().isoformat(),
            }

        context_data = self.conversation_contexts[conversation_id]
        turn_number = len(context_data["turns"]) + 1

        # Add current turn
        current_turn = {
            "turn": turn_number,
            "user_input": user_input,
            "timestamp": datetime.now().isoformat(),
            "analysis": await self.analyze_text(user_input, {"domain": domain}),
        }

        context_data["turns"].append(current_turn)

        # Limit conversation history (keep last 10 turns)
        if len(context_data["turns"]) > 10:
            context_data["turns"] = context_data["turns"][-10:]

        # Create conversation context
        conversation_context = ConversationContext(
            conversation_id=conversation_id,
            turn_number=turn_number,
            previous_turns=context_data["turns"][:-1],  # All but current turn
            user_profile=user_profile,
            domain=domain,
            session_state=context_data["session_state"],
        )

        return conversation_context

    async def detect_toxicity(self, text: str) -> Dict[str, Any]:
        """Detect toxicity and harmful content."""
        toxicity_result = {
            "is_toxic": False,
            "toxicity_score": 0.0,
            "categories": [],
            "confidence": 0.0,
        }

        try:
            if "toxicity" in self.models:
                result = self.models["toxicity"](text)
                toxicity_result["is_toxic"] = result[0]["label"] == "TOXIC"
                toxicity_result["toxicity_score"] = float(result[0]["score"])
                toxicity_result["confidence"] = float(result[0]["score"])

            else:
                # Fallback: simple keyword-based detection
                toxic_keywords = [
                    "hate",
                    "kill",
                    "die",
                    "stupid",
                    "idiot",
                    "moron",
                    "dumb",
                    "shut up",
                    "fuck",
                    "shit",
                    "damn",
                    "hell",
                ]

                text_lower = text.lower()
                toxic_matches = [
                    keyword for keyword in toxic_keywords if keyword in text_lower
                ]

                if toxic_matches:
                    toxicity_result["is_toxic"] = True
                    toxicity_result["toxicity_score"] = min(
                        1.0, len(toxic_matches) / 10.0
                    )
                    toxicity_result["categories"] = toxic_matches
                    toxicity_result["confidence"] = 0.6

        except Exception as e:
            logger.warning(f"Toxicity detection error: {e}")

        return toxicity_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        success_rate = 0.0
        if self.performance_metrics["total_requests"] > 0:
            success_rate = (
                self.performance_metrics["successful_requests"]
                / self.performance_metrics["total_requests"]
            )

        return {
            "specialist_id": self.specialist_id,
            "version": self.version,
            "capabilities": self.capabilities,
            "models_loaded": len(self.models),
            "advanced_nlp_available": ADVANCED_NLP_AVAILABLE,
            "performance": {**self.performance_metrics, "success_rate": success_rate},
            "active_conversations": len(self.conversation_contexts),
        }

    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversation contexts."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        conversations_to_remove = []
        for conv_id, context in self.conversation_contexts.items():
            created_at = datetime.fromisoformat(context["created_at"]).timestamp()
            if created_at < cutoff_time:
                conversations_to_remove.append(conv_id)

        for conv_id in conversations_to_remove:
            del self.conversation_contexts[conv_id]

        logger.info(
            f"Cleaned up {len(conversations_to_remove)} old conversation contexts"
        )

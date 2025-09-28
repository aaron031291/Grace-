"""Enhanced W5H Indexer - Elite-level NLP processing for Who/What/When/Where/Why/How indexing."""
import re
import spacy
import nltk
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import logging

from ..contracts.dto_common import W5HIndex, MemoryEntry

logger = logging.getLogger(__name__)


class EnhancedW5HIndexer:
    """Elite-level NLP W5H extraction with advanced language understanding."""
    
    def __init__(self):
        """Initialize with advanced NLP models."""
        self._initialize_models()
        self._initialize_patterns()
        
    def _initialize_models(self):
        """Initialize advanced NLP models."""
        try:
            # Load spaCy model for advanced NLP
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Initialize question-answering for context understanding
            self.qa_pipeline = pipeline("question-answering", 
                                      model="distilbert-base-uncased-distilled-squad")
            
            # Initialize named entity recognition
            self.ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
            
            logger.info("Enhanced W5H Indexer models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize advanced models: {e}. Falling back to basic patterns.")
            self.nlp = None
            self.sentiment_analyzer = None
            self.qa_pipeline = None
            self.ner_pipeline = None
    
    def _initialize_patterns(self):
        """Initialize pattern matching for basic extraction."""
        # Enhanced WHO patterns
        self.who_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Full names
            r'\b([A-Z]\. [A-Z][a-z]+)\b',  # Initials and last name
            r'\b(@\w+)\b',  # Social media handles
            r'\b(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+([A-Z][a-z]+)\b',  # Titles with names
            r'\b(user|admin|system|operator|manager|supervisor|director|CEO|CTO|developer|engineer|analyst)\b',  # Roles
            r'\b(team|department|group|committee|board)\b',  # Groups
        ]
        
        # Enhanced WHAT patterns
        self.what_patterns = [
            r'\b(creat|generat|produc|build|construct|develop|design|implement|deploy)\w*\b',  # Creation actions
            r'\b(updat|modif|chang|edit|alter|revise|refactor)\w*\b',  # Modification actions
            r'\b(delet|remov|destroy|eliminat|purge|clear)\w*\b',  # Deletion actions
            r'\b(analyz|examin|investigat|stud|review|audit|inspect|evaluat)\w*\b',  # Analysis actions
            r'\b(process|handl|manag|execut|run|perform|conduct)\w*\b',  # Processing actions
            r'\b(document|file|record|entry|data|information|content|report|log)\b',  # Objects
            r'\b(system|application|service|platform|database|server|network)\b',  # Tech objects
            r'\b(task|project|initiative|program|workflow|procedure|process)\b',  # Work objects
        ]
        
        # Enhanced WHEN patterns
        self.when_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',  # Dates
            r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\b',  # Times
            r'\b(today|tomorrow|yesterday|now|currently|recently|soon|later)\b',  # Temporal references
            r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Days of week
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Months
            r'\b(morning|afternoon|evening|night|dawn|dusk|noon|midnight)\b',  # Time periods
            r'\b(before|after|during|while|when|until|since|from|to)\s+([^.!?]+)\b',  # Temporal relationships
        ]
        
        # Enhanced WHERE patterns
        self.where_patterns = [
            r'\b(server|database|system|local|remote|cloud|datacenter|office|building)\b',  # Locations
            r'\b([a-z]+://[\w\.\/\-\?\=\&\%\#]+)\b',  # URLs
            r'\b(\w+\.\w{2,4})\b',  # Domains
            r'\b([A-Z][a-z]+,\s*[A-Z][a-z]*)\b',  # City, State
            r'\b(online|offline|on-site|off-site|remotely|locally)\b',  # Virtual/physical locations
            r'\b(room|floor|level|section|area|zone|region)\s+(\w+)\b',  # Physical locations
        ]
        
        # Enhanced WHY patterns
        self.why_patterns = [
            r'\bbecause\s+([^.!?]+)',  # Causal reasons
            r'\bfor\s+([^.!?]+)',  # Purposes
            r'\bin order to\s+([^.!?]+)',  # Intentions
            r'\bso that\s+([^.!?]+)',  # Goals
            r'\bdue to\s+([^.!?]+)',  # Causes
            r'\bas a result of\s+([^.!?]+)',  # Consequences
            r'\bto\s+(achieve|accomplish|ensure|prevent|avoid|improve|enhance|fix|resolve)\s+([^.!?]+)',  # Objectives
        ]
        
        # Enhanced HOW patterns
        self.how_patterns = [
            r'\busing\s+([^.!?]+)',  # Methods/tools
            r'\bvia\s+([^.!?]+)',  # Mechanisms
            r'\bthrough\s+([^.!?]+)',  # Processes
            r'\bby\s+([^.!?]+)',  # Means
            r'\bwith\s+([^.!?]+)',  # Instruments
            r'\bby means of\s+([^.!?]+)',  # Methods
            r'\bstep by step\s+([^.!?]+)',  # Procedures
        ]
    
    def extract(self, content: str, context: Optional[Dict[str, Any]] = None) -> W5HIndex:
        """Extract W5H elements using advanced NLP techniques."""
        index = W5HIndex()
        
        # Preprocess content
        cleaned_content = self._preprocess_content(content)
        
        # Use advanced NLP if available
        if self.nlp is not None:
            index = self._extract_with_advanced_nlp(cleaned_content, context)
        else:
            index = self._extract_with_patterns(cleaned_content)
        
        # Post-process and enhance results
        index = self._enhance_extraction(index, cleaned_content, context)
        
        return index
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for better extraction."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Handle common contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            content = content.replace(contraction, expansion)
        
        return content
    
    def _extract_with_advanced_nlp(self, content: str, context: Optional[Dict[str, Any]]) -> W5HIndex:
        """Extract W5H elements using advanced NLP models."""
        index = W5HIndex()
        
        try:
            # Process with spaCy
            doc = self.nlp(content)
            
            # Extract WHO using NER and dependency parsing
            index.who = self._extract_who_advanced(doc, content)
            
            # Extract WHAT using verb-object relations and semantic understanding
            index.what = self._extract_what_advanced(doc, content)
            
            # Extract WHEN using temporal NER and date parsing
            index.when = self._extract_when_advanced(doc, content)
            
            # Extract WHERE using location NER and geographic understanding
            index.where = self._extract_where_advanced(doc, content)
            
            # Extract WHY using causal reasoning and sentiment analysis
            index.why = self._extract_why_advanced(doc, content)
            
            # Extract HOW using method extraction and procedural understanding
            index.how = self._extract_how_advanced(doc, content)
            
            # Add semantic understanding
            index = self._add_semantic_understanding(index, doc, content, context)
            
        except Exception as e:
            logger.error(f"Error in advanced NLP extraction: {e}")
            # Fallback to pattern-based extraction
            index = self._extract_with_patterns(content)
        
        return index
    
    def _extract_who_advanced(self, doc, content: str) -> List[str]:
        """Extract WHO entities using advanced NER and linguistic analysis."""
        who_entities = []
        
        # Extract people using NER
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(content)
                for entity in ner_results:
                    if entity['entity_group'] in ['PER', 'PERSON']:
                        who_entities.append(entity['word'])
            except Exception as e:
                logger.warning(f"NER pipeline error: {e}")
        
        # Extract using spaCy's named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                who_entities.append(ent.text)
        
        # Extract roles and titles using dependency parsing
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'nsubjpass']:
                # Check if it's likely a person or role
                if any(role in token.text.lower() for role in ['user', 'admin', 'manager', 'developer', 'analyst']):
                    who_entities.append(token.text)
        
        # Pattern-based extraction as fallback
        for pattern in self.who_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            who_entities.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return list(set([entity.strip() for entity in who_entities if entity.strip()]))
    
    def _extract_what_advanced(self, doc, content: str) -> List[str]:
        """Extract WHAT actions and objects using semantic analysis."""
        what_entities = []
        
        # Extract main verbs and their objects
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Get the verb lemma
                what_entities.append(token.lemma_)
                
                # Get direct objects
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr']:
                        what_entities.append(child.text)
        
        # Extract noun phrases as potential objects
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                what_entities.append(chunk.text)
        
        # Pattern-based extraction for actions
        for pattern in self.what_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            what_entities.extend(matches)
        
        return list(set([entity.strip() for entity in what_entities if entity.strip()]))
    
    def _extract_when_advanced(self, doc, content: str) -> List[str]:
        """Extract WHEN temporal entities using advanced date/time understanding."""
        when_entities = []
        
        # Extract temporal entities using spaCy
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME', 'CARDINAL']:
                # Validate if it's actually temporal
                if self._is_temporal_entity(ent.text):
                    when_entities.append(ent.text)
        
        # Pattern-based extraction
        for pattern in self.when_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            when_entities.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return list(set([entity.strip() for entity in when_entities if entity.strip()]))
    
    def _extract_where_advanced(self, doc, content: str) -> List[str]:
        """Extract WHERE location entities using geographic NLP."""
        where_entities = []
        
        # Extract locations using spaCy NER
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical, location, facility
                where_entities.append(ent.text)
        
        # Pattern-based extraction
        for pattern in self.where_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            where_entities.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return list(set([entity.strip() for entity in where_entities if entity.strip()]))
    
    def _extract_why_advanced(self, doc, content: str) -> List[str]:
        """Extract WHY reasons using causal analysis and sentiment understanding."""
        why_entities = []
        
        # Analyze sentiment to understand motivation
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(content[:500])  # Limit length for processing
                sentiment_score = sentiment[0]['label']
                confidence = sentiment[0]['score']
                
                # Add sentiment-based reasoning
                if confidence > 0.8:
                    why_entities.append(f"motivated by {sentiment_score.lower()} sentiment")
            except Exception as e:
                logger.warning(f"Sentiment analysis error: {e}")
        
        # Extract causal relationships using dependency parsing
        for token in doc:
            if token.text.lower() in ['because', 'since', 'due', 'for']:
                # Get the clause following the causal connector
                causal_clause = self._extract_dependent_clause(token, doc)
                if causal_clause:
                    why_entities.append(causal_clause)
        
        # Pattern-based extraction
        for pattern in self.why_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            why_entities.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([entity.strip() for entity in why_entities if entity.strip()]))
    
    def _extract_how_advanced(self, doc, content: str) -> List[str]:
        """Extract HOW methods using procedural understanding."""
        how_entities = []
        
        # Extract method indicators using dependency parsing
        for token in doc:
            if token.text.lower() in ['using', 'via', 'through', 'by', 'with']:
                method_clause = self._extract_dependent_clause(token, doc)
                if method_clause:
                    how_entities.append(method_clause)
        
        # Extract tool mentions
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG'] and any(tool in ent.text.lower() 
                                                      for tool in ['api', 'system', 'tool', 'service', 'platform']):
                how_entities.append(f"using {ent.text}")
        
        # Pattern-based extraction
        for pattern in self.how_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            how_entities.extend([match.strip() for match in matches if match.strip()])
        
        return list(set([entity.strip() for entity in how_entities if entity.strip()]))
    
    def _extract_with_patterns(self, content: str) -> W5HIndex:
        """Fallback pattern-based extraction."""
        index = W5HIndex()
        
        # WHO - Extract people, entities, roles
        for pattern in self.who_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.who.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # WHAT - Extract actions, objects, subjects
        for pattern in self.what_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.what.extend(matches)
        
        # WHEN - Extract temporal references
        for pattern in self.when_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.when.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # WHERE - Extract locations, systems, places
        for pattern in self.where_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.where.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # WHY - Extract reasons, purposes, motivations
        for pattern in self.why_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.why.extend([match.strip() for match in matches if match.strip()])
        
        # HOW - Extract methods, tools, processes
        for pattern in self.how_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            index.how.extend([match.strip() for match in matches if match.strip()])
        
        # Remove duplicates
        index.who = list(set(index.who))
        index.what = list(set(index.what))
        index.when = list(set(index.when))
        index.where = list(set(index.where))
        index.why = list(set(index.why))
        index.how = list(set(index.how))
        
        return index
    
    def _add_semantic_understanding(self, index: W5HIndex, doc, content: str, 
                                  context: Optional[Dict[str, Any]]) -> W5HIndex:
        """Add semantic understanding and contextual information."""
        
        # Add TextBlob sentiment and subjectivity
        try:
            blob = TextBlob(content)
            sentiment = blob.sentiment
            
            # Add sentiment-driven insights
            if sentiment.polarity > 0.1:
                index.why.append("positive sentiment indicates favorable conditions")
            elif sentiment.polarity < -0.1:
                index.why.append("negative sentiment indicates challenges or issues")
            
            if sentiment.subjectivity > 0.7:
                index.why.append("highly subjective content indicates personal opinions")
            
        except Exception as e:
            logger.warning(f"TextBlob analysis error: {e}")
        
        # Add context-based enhancements
        if context:
            if 'domain' in context:
                domain = context['domain']
                index.where.append(f"within {domain} domain")
            
            if 'user_intent' in context:
                intent = context['user_intent']
                index.why.append(f"to fulfill user intent: {intent}")
        
        # Add linguistic features
        index = self._add_linguistic_features(index, doc)
        
        return index
    
    def _add_linguistic_features(self, index: W5HIndex, doc) -> W5HIndex:
        """Add advanced linguistic features to the index."""
        
        # Extract key phrases and concepts
        key_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                key_phrases.append(chunk.text)
        
        if key_phrases:
            index.what.extend(key_phrases[:5])  # Top 5 key phrases
        
        # Extract relationships between entities
        relationships = []
        for token in doc:
            if token.dep_ in ['prep', 'agent'] and token.head.pos_ == 'VERB':
                relationships.append(f"{token.head.text} {token.text}")
        
        if relationships:
            index.how.extend(relationships[:3])  # Top 3 relationships
        
        return index
    
    def _is_temporal_entity(self, text: str) -> bool:
        """Check if an entity is temporal."""
        temporal_keywords = [
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'today', 'tomorrow', 'yesterday', 'now', 'later', 'soon',
            'morning', 'afternoon', 'evening', 'night'
        ]
        
        text_lower = text.lower()
        return (
            any(keyword in text_lower for keyword in temporal_keywords) or
            re.search(r'\d{1,4}[/\-:]\d{1,2}', text) is not None
        )
    
    def _extract_dependent_clause(self, token, doc) -> Optional[str]:
        """Extract the dependent clause following a token."""
        clause_tokens = []
        
        # Get all tokens that depend on this token or follow it
        start_collecting = False
        for t in doc:
            if t == token:
                start_collecting = True
                continue
            
            if start_collecting:
                if t.text in ['.', '!', '?', ';']:
                    break
                clause_tokens.append(t.text)
        
        if clause_tokens:
            return ' '.join(clause_tokens[:15])  # Limit to 15 tokens
        
        return None
    
    def analyze_intent(self, content: str) -> Dict[str, Any]:
        """Analyze user intent using advanced NLP."""
        intent_analysis = {
            'primary_intent': 'unknown',
            'confidence': 0.0,
            'intent_type': 'informational',
            'entities': [],
            'sentiment': 'neutral'
        }
        
        if not self.nlp:
            return intent_analysis
        
        try:
            doc = self.nlp(content)
            
            # Classify intent type based on linguistic patterns
            question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
            action_verbs = ['create', 'update', 'delete', 'find', 'search', 'analyze', 'process']
            
            content_lower = content.lower()
            
            if any(word in content_lower for word in question_words):
                intent_analysis['intent_type'] = 'informational'
                intent_analysis['primary_intent'] = 'question'
                intent_analysis['confidence'] = 0.8
            elif any(verb in content_lower for verb in action_verbs):
                intent_analysis['intent_type'] = 'transactional'
                intent_analysis['primary_intent'] = 'action_request'
                intent_analysis['confidence'] = 0.7
            
            # Extract entities for intent understanding
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'confidence': 1.0
                })
            
            intent_analysis['entities'] = entities
            
            # Add sentiment
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(content[:500])
                intent_analysis['sentiment'] = sentiment_result[0]['label'].lower()
            
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
        
        return intent_analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced indexer statistics."""
        return {
            'version': '2.0.0-elite',
            'models_loaded': {
                'spacy': self.nlp is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'qa_pipeline': self.qa_pipeline is not None,
                'ner_pipeline': self.ner_pipeline is not None
            },
            'capabilities': [
                'advanced_ner',
                'sentiment_analysis',
                'intent_detection',
                'causal_reasoning',
                'temporal_understanding',
                'semantic_extraction',
                'linguistic_analysis'
            ]
        }
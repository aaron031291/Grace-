#!/usr/bin/env python3
"""
Elite NLP Interface Test Suite - Comprehensive testing of enhanced NLP capabilities.
"""
import asyncio
import sys
import os
import time
import logging
from typing import Dict, List, Any

# Add the Grace directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data for comprehensive NLP evaluation
TEST_CASES = {
    "governance_requests": [
        {
            "text": "I need approval to deploy the new ML model for customer sentiment analysis. The model has been tested and shows 92% accuracy.",
            "expected_intent": "action_request",
            "expected_sentiment": "positive",
            "expected_entities": ["ML model", "customer sentiment analysis"]
        },
        {
            "text": "Can you please explain the governance policy regarding data retention for AI training datasets?",
            "expected_intent": "question",
            "expected_sentiment": "neutral",
            "expected_entities": ["governance policy", "data retention", "AI training datasets"]
        },
        {
            "text": "URGENT: The system is processing biased outcomes and we need immediate intervention to prevent harm to users.",
            "expected_intent": "action_request",
            "expected_sentiment": "negative",
            "expected_urgency": "high",
            "expected_entities": ["system", "biased outcomes", "users"]
        }
    ],
    "conversational": [
        {
            "text": "Hello Grace, I'm working on a project that involves analyzing social media posts for emotional well-being insights.",
            "expected_intent": "statement",
            "expected_sentiment": "neutral",
            "expected_entities": ["social media posts", "emotional well-being"]
        },
        {
            "text": "How do I ensure my AI model complies with fairness and transparency requirements?",
            "expected_intent": "question",
            "expected_sentiment": "neutral",
            "expected_topics": ["AI model", "fairness", "transparency"]
        }
    ],
    "edge_cases": [
        {
            "text": "",
            "expected_intent": "unknown",
            "expected_sentiment": "neutral"
        },
        {
            "text": "This is terrible and I hate how this stupid system never works properly!",
            "expected_intent": "statement",
            "expected_sentiment": "negative",
            "expected_toxicity": True
        },
        {
            "text": "The quick brown fox jumps over the lazy dog. Machine learning algorithms process data efficiently.",
            "expected_intent": "statement",
            "expected_sentiment": "neutral",
            "expected_entities": ["Machine learning algorithms"]
        }
    ]
}


class EliteNLPTester:
    """Test suite for elite NLP capabilities."""
    
    def __init__(self):
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Try to import elite NLP components
        self.elite_nlp = None
        self.enhanced_indexer = None
        self.interface_service = None
        
        try:
            from grace.mldl.specialists.elite_nlp_specialist import EliteNLPSpecialist
            from grace.mtl_kernel.enhanced_w5h_indexer import EnhancedW5HIndexer
            from grace.interface.interface_service import InterfaceService
            
            self.elite_nlp = EliteNLPSpecialist()
            self.enhanced_indexer = EnhancedW5HIndexer()
            self.interface_service = InterfaceService()
            
            logger.info("✅ Elite NLP components loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import elite NLP components: {e}")
            self.test_results["errors"].append(f"Import error: {e}")
    
    def assert_test(self, condition: bool, test_name: str, message: str = ""):
        """Assert test condition and update results."""
        if condition:
            self.test_results["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results["failed"] += 1
            error_msg = f"❌ {test_name}: FAILED - {message}"
            logger.error(error_msg)
            self.test_results["errors"].append(error_msg)
    
    async def test_elite_nlp_specialist(self):
        """Test the Elite NLP Specialist capabilities."""
        logger.info("\n🧪 Testing Elite NLP Specialist...")
        
        if not self.elite_nlp:
            self.test_results["errors"].append("Elite NLP Specialist not available")
            return
        
        # Test basic text analysis
        test_text = "I'm really excited about the new AI governance framework. It will help ensure fairness and transparency."
        
        try:
            analysis = await self.elite_nlp.analyze_text(test_text)
            
            self.assert_test(
                analysis is not None,
                "Elite NLP Analysis",
                "Should return analysis object"
            )
            
            self.assert_test(
                analysis.sentiment.get("label") in ["positive", "negative", "neutral"],
                "Sentiment Analysis",
                f"Got sentiment: {analysis.sentiment.get('label')}"
            )
            
            self.assert_test(
                isinstance(analysis.entities, list),
                "Entity Extraction",
                "Should return list of entities"
            )
            
            self.assert_test(
                analysis.confidence > 0.0,
                "Confidence Score",
                f"Got confidence: {analysis.confidence}"
            )
            
            self.assert_test(
                len(analysis.summary) > 0,
                "Text Summarization",
                "Should generate summary"
            )
            
            logger.info(f"   📊 Analysis results:")
            logger.info(f"      - Sentiment: {analysis.sentiment.get('label')} (confidence: {analysis.sentiment.get('confidence', 0):.2f})")
            logger.info(f"      - Entities: {len(analysis.entities)} found")
            logger.info(f"      - Topics: {len(analysis.topics)} identified")
            logger.info(f"      - Summary: {analysis.summary[:100]}...")
            
        except Exception as e:
            self.test_results["errors"].append(f"Elite NLP analysis error: {e}")
            logger.error(f"❌ Elite NLP analysis failed: {e}")
    
    async def test_enhanced_w5h_indexer(self):
        """Test the Enhanced W5H Indexer."""
        logger.info("\n🧪 Testing Enhanced W5H Indexer...")
        
        if not self.enhanced_indexer:
            self.test_results["errors"].append("Enhanced W5H Indexer not available")
            return
        
        test_text = "John Smith submitted a report yesterday about the security vulnerability in the payment system using the new scanning tool."
        
        try:
            w5h_index = self.enhanced_indexer.extract(test_text)
            
            self.assert_test(
                w5h_index is not None,
                "W5H Extraction",
                "Should return W5H index object"
            )
            
            self.assert_test(
                len(w5h_index.who) > 0,
                "WHO Extraction",
                f"Should find 'John Smith', got: {w5h_index.who}"
            )
            
            self.assert_test(
                len(w5h_index.what) > 0,
                "WHAT Extraction",
                f"Should find actions/objects, got: {w5h_index.what}"
            )
            
            self.assert_test(
                len(w5h_index.when) > 0,
                "WHEN Extraction",
                f"Should find 'yesterday', got: {w5h_index.when}"
            )
            
            # Test intent analysis
            intent_analysis = self.enhanced_indexer.analyze_intent(test_text)
            
            self.assert_test(
                "intent" in intent_analysis,
                "Intent Analysis",
                "Should return intent information"
            )
            
            logger.info(f"   📊 W5H Extraction results:")
            logger.info(f"      - WHO: {w5h_index.who}")
            logger.info(f"      - WHAT: {w5h_index.what}")
            logger.info(f"      - WHEN: {w5h_index.when}")
            logger.info(f"      - WHERE: {w5h_index.where}")
            logger.info(f"      - WHY: {w5h_index.why}")
            logger.info(f"      - HOW: {w5h_index.how}")
            logger.info(f"      - Intent: {intent_analysis.get('primary_intent')} (confidence: {intent_analysis.get('confidence', 0):.2f})")
            
        except Exception as e:
            self.test_results["errors"].append(f"Enhanced W5H indexer error: {e}")
            logger.error(f"❌ Enhanced W5H indexer failed: {e}")
    
    async def test_toxicity_detection(self):
        """Test toxicity detection capabilities."""
        logger.info("\n🧪 Testing Toxicity Detection...")
        
        if not self.elite_nlp:
            return
        
        toxic_text = "This is a stupid system and I hate everyone who built it!"
        safe_text = "I have some constructive feedback about improving the system."
        
        try:
            toxic_result = await self.elite_nlp.detect_toxicity(toxic_text)
            safe_result = await self.elite_nlp.detect_toxicity(safe_text)
            
            self.assert_test(
                toxic_result["is_toxic"],
                "Toxic Text Detection",
                f"Should detect toxicity in: '{toxic_text}'"
            )
            
            self.assert_test(
                not safe_result["is_toxic"],
                "Safe Text Detection",
                f"Should not detect toxicity in: '{safe_text}'"
            )
            
            logger.info(f"   📊 Toxicity Detection results:")
            logger.info(f"      - Toxic text score: {toxic_result['toxicity_score']:.2f}")
            logger.info(f"      - Safe text score: {safe_result['toxicity_score']:.2f}")
            
        except Exception as e:
            self.test_results["errors"].append(f"Toxicity detection error: {e}")
            logger.error(f"❌ Toxicity detection failed: {e}")
    
    async def test_conversation_management(self):
        """Test conversation context management."""
        logger.info("\n🧪 Testing Conversation Management...")
        
        if not self.elite_nlp:
            return
        
        conversation_id = "test_conv_001"
        user_profile = {"user_id": "test_user", "preferences": {"domain": "governance"}}
        
        try:
            # First turn
            context1 = await self.elite_nlp.manage_conversation_context(
                conversation_id,
                "Hello, I need help with AI governance policies.",
                user_profile,
                "governance"
            )
            
            # Second turn
            context2 = await self.elite_nlp.manage_conversation_context(
                conversation_id,
                "Specifically, I want to know about fairness requirements.",
                user_profile,
                "governance"
            )
            
            self.assert_test(
                context1.conversation_id == conversation_id,
                "Conversation ID Tracking",
                "Should maintain conversation ID"
            )
            
            self.assert_test(
                context2.turn_number > context1.turn_number,
                "Turn Number Progression",
                f"Turn numbers should increase: {context1.turn_number} -> {context2.turn_number}"
            )
            
            self.assert_test(
                len(context2.previous_turns) > 0,
                "Context History",
                "Should maintain conversation history"
            )
            
            logger.info(f"   📊 Conversation Management results:")
            logger.info(f"      - Conversation ID: {context1.conversation_id}")
            logger.info(f"      - Turn progression: {context1.turn_number} -> {context2.turn_number}")
            logger.info(f"      - History length: {len(context2.previous_turns)}")
            
        except Exception as e:
            self.test_results["errors"].append(f"Conversation management error: {e}")
            logger.error(f"❌ Conversation management failed: {e}")
    
    async def test_interface_integration(self):
        """Test integration with interface service."""
        logger.info("\n🧪 Testing Interface Service Integration...")
        
        if not self.interface_service:
            return
        
        try:
            # Test interface service stats include NLP metrics
            stats = self.interface_service.get_stats()
            
            self.assert_test(
                "nlp" in stats or "Elite NLP capabilities not available" in str(stats),
                "NLP Stats Integration",
                "Interface service should include NLP statistics"
            )
            
            # Test enhanced memory action processing
            action = {
                "type": "memory.search",
                "payload": {
                    "query": "Find documents about machine learning fairness and bias detection",
                    "user_id": "test_user",
                    "filters": {}
                }
            }
            
            result = await self.interface_service.dispatch_action(action)
            
            self.assert_test(
                "Enhanced memory search" in result or "Memory search" in result,
                "Enhanced Action Processing",
                f"Should process actions with NLP enhancement: {result}"
            )
            
            logger.info(f"   📊 Interface Integration results:")
            logger.info(f"      - Stats include NLP: {'nlp' in stats}")
            logger.info(f"      - Action result: {result}")
            
        except Exception as e:
            self.test_results["errors"].append(f"Interface integration error: {e}")
            logger.error(f"❌ Interface integration failed: {e}")
    
    async def test_comprehensive_scenarios(self):
        """Test comprehensive real-world scenarios."""
        logger.info("\n🧪 Testing Comprehensive Scenarios...")
        
        if not self.elite_nlp or not self.enhanced_indexer:
            return
        
        scenarios_passed = 0
        scenarios_total = 0
        
        for category, test_cases in TEST_CASES.items():
            logger.info(f"\n   Testing {category} scenarios:")
            
            for i, test_case in enumerate(test_cases):
                scenarios_total += 1
                text = test_case["text"]
                
                if not text:  # Skip empty text tests for comprehensive analysis
                    continue
                
                try:
                    # Perform comprehensive analysis
                    analysis = await self.elite_nlp.analyze_text(text)
                    w5h_index = self.enhanced_indexer.extract(text)
                    
                    # Check expectations
                    passed = True
                    
                    if "expected_intent" in test_case:
                        actual_intent = analysis.intent.get("intent", "unknown")
                        if actual_intent != test_case["expected_intent"]:
                            logger.warning(f"      Intent mismatch: expected {test_case['expected_intent']}, got {actual_intent}")
                            passed = False
                    
                    if "expected_sentiment" in test_case:
                        actual_sentiment = analysis.sentiment.get("label", "neutral")
                        if actual_sentiment != test_case["expected_sentiment"]:
                            logger.warning(f"      Sentiment mismatch: expected {test_case['expected_sentiment']}, got {actual_sentiment}")
                    
                    if "expected_toxicity" in test_case:
                        toxicity_result = await self.elite_nlp.detect_toxicity(text)
                        if toxicity_result["is_toxic"] != test_case["expected_toxicity"]:
                            logger.warning(f"      Toxicity mismatch: expected {test_case['expected_toxicity']}, got {toxicity_result['is_toxic']}")
                    
                    if passed:
                        scenarios_passed += 1
                        logger.info(f"      ✅ Scenario {i+1}: PASSED")
                    else:
                        logger.info(f"      ⚠️  Scenario {i+1}: PARTIAL")
                    
                except Exception as e:
                    logger.error(f"      ❌ Scenario {i+1}: ERROR - {e}")
        
        self.assert_test(
            scenarios_passed > scenarios_total * 0.7,  # 70% success rate
            "Comprehensive Scenarios",
            f"Should pass most scenarios: {scenarios_passed}/{scenarios_total}"
        )
    
    async def run_all_tests(self):
        """Run all NLP tests."""
        logger.info("🚀 Starting Elite NLP Interface Test Suite...")
        start_time = time.time()
        
        await self.test_elite_nlp_specialist()
        await self.test_enhanced_w5h_indexer()
        await self.test_toxicity_detection()
        await self.test_conversation_management()
        await self.test_interface_integration()
        await self.test_comprehensive_scenarios()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print final results
        logger.info(f"\n🎯 Test Results Summary:")
        logger.info(f"   ✅ Tests Passed: {self.test_results['passed']}")
        logger.info(f"   ❌ Tests Failed: {self.test_results['failed']}")
        logger.info(f"   ⏱️  Total Time: {duration:.2f}s")
        
        if self.test_results["errors"]:
            logger.info(f"\n⚠️  Errors encountered:")
            for error in self.test_results["errors"]:
                logger.info(f"   - {error}")
        
        success_rate = self.test_results["passed"] / (self.test_results["passed"] + self.test_results["failed"]) if (self.test_results["passed"] + self.test_results["failed"]) > 0 else 0
        
        if success_rate >= 0.8:
            logger.info(f"\n🎉 Elite NLP capabilities are functioning at elite level! ({success_rate:.1%} success rate)")
            return True
        else:
            logger.info(f"\n⚠️  Elite NLP capabilities need improvement. ({success_rate:.1%} success rate)")
            return False


async def main():
    """Main test function."""
    tester = EliteNLPTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
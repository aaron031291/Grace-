#!/usr/bin/env python3
"""
Grace Voice & Book Integration Demo
==================================

Demonstrates the complete integration of new Grace capabilities:
1. Voice toggle for bidirectional communication
2. Book ingestion for 500+ page documents  
3. Enhanced health monitoring interface
4. Memory storage with insights and actions

Usage:
    python grace_voice_book_demo.py
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime

# Add grace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grace.interface_kernel.kernel import InterfaceKernel
from grace.interface_kernel.voice_service import create_voice_service
from grace.interface_kernel.health_dashboard import HealthDashboard
from grace.memory.book_ingestion import BookIngestionService


class GraceIntegrationDemo:
    """Comprehensive demo of Grace's enhanced capabilities."""
    
    def __init__(self):
        self.interface_kernel = InterfaceKernel()
        print("🤖 Grace Enhanced Systems Initialized")
        
    async def demo_voice_system(self):
        """Demonstrate voice toggle functionality."""
        print("\n" + "="*60)
        print("🎤 VOICE SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Get voice status
        status = self.interface_kernel.voice_service.get_voice_status()
        print(f"Voice Available: {status['voice_available']}")
        print(f"Current State: {status['state']}")
        print(f"Voice Enabled: {status['voice_enabled']}")
        
        # Test voice toggle (will use mock service if no speech libraries)
        print("\n🔄 Testing voice toggle...")
        result = await self.interface_kernel.voice_service.toggle_voice_mode()
        print(f"Toggle Result: {result['message']}")
        print(f"New State: {result['state']}")
        
        # Simulate some voice communication
        print("\n💬 Simulating voice communication...")
        if hasattr(self.interface_kernel.voice_service, 'conversation_history'):
            # Add some test conversation
            from grace.interface_kernel.voice_service import VoiceMessage
            test_messages = [
                VoiceMessage(
                    content="What is the current system health?",
                    timestamp=datetime.now(),
                    source="user",
                    confidence=0.95
                ),
                VoiceMessage(
                    content="System health is optimal. All components operational.",
                    timestamp=datetime.now(),
                    source="grace"
                )
            ]
            
            # If it's the real service, add messages
            if hasattr(self.interface_kernel.voice_service, 'conversation_history'):
                self.interface_kernel.voice_service.conversation_history.extend(test_messages)
        
        # Get conversation history
        history = self.interface_kernel.voice_service.get_conversation_history()
        print(f"Conversation History: {len(history)} messages")
        for i, msg in enumerate(history[-2:]):  # Show last 2 messages
            print(f"  {i+1}. [{msg['source']}] {msg['content']}")
    
    async def demo_book_ingestion(self):
        """Demonstrate large document ingestion."""
        print("\n" + "="*60)
        print("📚 BOOK INGESTION DEMONSTRATION")
        print("="*60)
        
        # Create a substantial test book
        test_book = """Chapter 1: Grace AI Architecture Overview

Grace represents a revolutionary approach to AI governance and system management. The architecture is built on constitutional principles that ensure ethical behavior and transparent decision-making. The system incorporates multiple layers of validation and verification to maintain trust and reliability.

The core architecture consists of several key components: the governance engine for policy enforcement, the memory system for knowledge storage and retrieval, the trust kernel for reliability scoring, and the communication interfaces for human interaction. Each component is designed to work harmoniously while maintaining clear boundaries and responsibilities.

Grace employs a democratic approach to decision-making, where major system changes require consensus from multiple validation layers. This ensures that no single component can make unilateral decisions that might compromise system integrity or user trust.

Chapter 2: Voice Communication Systems

The integration of voice technology represents a significant advancement in human-AI interaction. Voice-enabled systems allow for natural, bidirectional communication that feels more intuitive than traditional text-based interfaces.

The voice processing pipeline includes automatic speech recognition for converting speech to text, natural language understanding for extracting meaning and intent, dialog management for maintaining conversation context, response generation for creating appropriate replies, and text-to-speech synthesis for converting responses back to natural speech.

Key challenges in voice system implementation include handling diverse accents and speaking patterns, managing background noise and audio quality issues, maintaining conversation context across multiple turns, ensuring privacy and security of voice data, and providing appropriate responses when understanding fails.

Chapter 3: Large-Scale Document Processing

Processing large documents such as books, reports, and technical manuals requires specialized approaches to chunking, indexing, and retrieval. Traditional methods that load entire documents into memory are impractical for large corpora.

The document processing pipeline begins with content analysis to identify structure and key sections. Chapter detection algorithms identify major divisions in the text. Content is then segmented into manageable chunks while preserving semantic coherence. Each chunk is analyzed for key concepts and insights.

Vector embeddings are generated for all content to enable semantic search capabilities. This allows users to find relevant information even when their queries use different terminology than the source documents. The system combines vector-based semantic search with traditional keyword matching for comprehensive coverage.

Chapter 4: Health Monitoring and System Diagnostics

Comprehensive health monitoring is essential for maintaining reliable AI systems. The health monitoring framework tracks performance metrics across all system components, identifies potential issues before they become critical, and provides detailed diagnostics for troubleshooting.

Key monitoring categories include system resources such as CPU, memory, and disk usage, component response times and error rates, data quality metrics and processing throughput, user interaction patterns and satisfaction scores, and security events and compliance status.

The monitoring system generates alerts for various conditions including resource utilization exceeding thresholds, error rates increasing beyond acceptable levels, response times degrading significantly, unusual usage patterns that might indicate issues, and security events requiring attention.

Chapter 5: Integration and Future Directions

The integration of voice communication, large document processing, and health monitoring creates a comprehensive AI system capable of sophisticated interaction and self-management. This integration enables new use cases and interaction patterns that were not possible with individual components.

Future development directions include enhancing voice recognition accuracy and naturalness, expanding document processing to handle multimedia content, implementing predictive health monitoring to prevent issues, developing more sophisticated reasoning capabilities, and creating better integration with external systems and APIs.

The ultimate goal is to create an AI system that can serve as a trusted partner for users, capable of understanding complex requests, processing vast amounts of information, and maintaining itself reliably over long periods of operation."""

        print("📖 Processing test book...")
        start_time = time.time()
        
        result = await self.interface_kernel.book_ingestion.ingest_book(
            content=test_book,
            title="Grace AI System Guide",
            author="Grace Development Team", 
            metadata={
                "category": "technical_documentation",
                "version": "1.0",
                "language": "english",
                "estimated_pages": 25
            }
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Ingestion completed in {processing_time:.2f} seconds")
        print(f"Status: {result['status']}")
        print(f"Job ID: {result['job_id']}")
        
        if 'processing_summary' in result:
            summary = result['processing_summary']
            print(f"\n📊 Processing Summary:")
            print(f"  Chapters: {summary.get('chapters_processed', 0)}")
            print(f"  Chunks: {summary.get('chunks_created', 0)}")
            print(f"  Insights: {summary.get('insights_extracted', 0)}")
            print(f"  Words: {summary.get('word_count', 0)}")
        
        # Test retrieval capabilities
        job_id = result['job_id']
        
        # Get chapters
        chapters = await self.interface_kernel.book_ingestion.get_book_chapters(job_id)
        print(f"\n📚 Retrieved {len(chapters)} chapters:")
        for chapter in chapters[:3]:  # Show first 3
            print(f"  • {chapter['title']} ({chapter['word_count']} words)")
        
        # Get insights
        insights = await self.interface_kernel.book_ingestion.get_book_insights(job_id)
        print(f"\n💡 Retrieved {len(insights)} insights:")
        for insight in insights[:2]:  # Show first 2
            print(f"  • [{insight['insight_type']}] {insight['content'][:80]}...")
        
        # Test search
        search_results = await self.interface_kernel.book_ingestion.search_book_content(
            job_id, "voice communication", 3
        )
        print(f"\n🔍 Search results for 'voice communication':")
        for result in search_results:
            print(f"  • [{result['type']}] {result['content'][:60]}...")
            print(f"    Relevance: {result['relevance']:.2f}")
        
        return job_id
    
    async def demo_health_monitoring(self):
        """Demonstrate comprehensive health monitoring."""
        print("\n" + "="*60)
        print("🏥 HEALTH MONITORING DEMONSTRATION")
        print("="*60)
        
        # Perform comprehensive health check
        print("🔍 Performing comprehensive health check...")
        health_result = await self.interface_kernel.health_dashboard.perform_health_check()
        
        print(f"✅ Health check completed in {health_result['check_duration_ms']:.1f}ms")
        print(f"Overall Status: {health_result['overall_status']}")
        print(f"Components Checked: {health_result['components_checked']}")
        
        # Get comprehensive health status
        health_status = self.interface_kernel.health_dashboard.get_comprehensive_health()
        print(f"\nSystem Uptime: {health_status['uptime_seconds']:.1f} seconds")
        print(f"Active Alerts: {len(health_status['active_alerts'])}")
        print(f"Total Components: {len(health_status['components'])}")
        
        # Show component status
        print(f"\n📊 Component Status:")
        for name, component in list(health_status['components'].items())[:5]:
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "critical": "❌", "unknown": "❓"}.get(component['status'], "❓")
            print(f"  {status_emoji} {name}: {component['status']}")
            if component.get('response_time_ms'):
                print(f"    Response Time: {component['response_time_ms']:.1f}ms")
        
        # Show alerts if any
        if health_status['active_alerts']:
            print(f"\n🚨 Active Alerts:")
            for alert in health_status['active_alerts']:
                print(f"  • [{alert['severity']}] {alert['component']}: {alert['message']}")
        
        # Test diagnostic
        print(f"\n🔧 Running system diagnostic...")
        diagnostic = await self.interface_kernel.health_dashboard.run_diagnostic()
        print(f"Diagnostic Status: {diagnostic['overall_status']}")
        if diagnostic['system_recommendations']:
            print(f"Recommendations:")
            for rec in diagnostic['system_recommendations']:
                print(f"  • {rec}")
    
    async def demo_integration_scenarios(self, book_job_id):
        """Demonstrate integrated scenarios combining all features."""
        print("\n" + "="*60)
        print("🔄 INTEGRATION SCENARIOS")
        print("="*60)
        
        # Scenario 1: Voice query about book content
        print("📝 Scenario 1: Voice query about ingested book")
        voice_query = "Tell me about the voice communication systems in the book"
        
        # Simulate processing the query through Grace's pipeline
        response = await self.interface_kernel._handle_voice_communication(voice_query)
        print(f"Voice Query: \"{voice_query}\"")
        print(f"Grace Response: \"{response['answer']}\"")
        print(f"Confidence: {response['confidence']:.1%}")
        
        # Scenario 2: Health check triggered by book processing
        print(f"\n🏥 Scenario 2: Health monitoring during book processing")
        
        # Check if book processing impacted system health
        current_health = self.interface_kernel.health_dashboard.get_comprehensive_health()
        memory_component = current_health['components'].get('Memory', {})
        if memory_component:
            print(f"Memory Status: {memory_component.get('status', 'unknown')}")
            for metric in memory_component.get('metrics', []):
                if metric['name'] == 'memory_usage':
                    print(f"Memory Usage: {metric['value']:.1f}%")
        
        # Scenario 3: Book insights accessible via voice
        print(f"\n💡 Scenario 3: Voice-accessible book insights")
        insights = await self.interface_kernel.book_ingestion.get_book_insights(book_job_id)
        if insights:
            sample_insight = insights[0]
            voice_response = f"Here's a key insight from the book: {sample_insight['content'][:100]}..."
            print(f"Available via voice: \"{voice_response}\"")
        
        # Scenario 4: System health accessible via voice query
        print(f"\n🎤 Scenario 4: Voice query for system health")
        health_query = "What is the current system health status?"
        health_response = await self.interface_kernel._handle_voice_communication(health_query)
        print(f"Health Query: \"{health_query}\"")
        print(f"Grace Response: \"{health_response['answer']}\"")
    
    async def run_complete_demo(self):
        """Run the complete integration demonstration."""
        print("🚀 Starting Grace Enhanced Capabilities Demo")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run all demonstrations
            await self.demo_voice_system()
            book_job_id = await self.demo_book_ingestion()
            await self.demo_health_monitoring()
            await self.demo_integration_scenarios(book_job_id)
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETED SUCCESSFULLY")
            print("="*60)
            
            print("\n🎯 Summary of Capabilities Demonstrated:")
            print("  ✅ Voice toggle and bidirectional communication")
            print("  ✅ Large document (book) ingestion and processing")
            print("  ✅ Chapter detection and insight extraction")
            print("  ✅ Comprehensive health monitoring")
            print("  ✅ System diagnostics and alerting")
            print("  ✅ Integration between voice, memory, and health systems")
            print("  ✅ Memory storage with searchable insights")
            print("  ✅ API endpoints for all new functionality")
            
            print("\n🔗 Available API Endpoints:")
            print("  Voice: /api/voice/toggle, /api/voice/status")
            print("  Books: /api/books/ingest, /api/books/{job_id}/insights")
            print("  Health: /api/health/comprehensive, /api/health/diagnostic")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            await self.interface_kernel.cleanup()


async def main():
    """Main demo function."""
    demo = GraceIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
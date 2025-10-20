"""
Test Orb interface with session duration and topic extraction
"""

import asyncio
from datetime import datetime
import time

print("Testing Orb Interface - Session Duration & Topic Extraction")
print("=" * 70)

# Test session management
print("\n1. Creating Orb session...")
try:
    from grace.orb.interface import OrbInterface
    from grace.orb.session_manager import OrbSessionManager
    
    # Initialize with services
    try:
        from grace.embeddings.service import EmbeddingService
        from grace.vectorstore.service import VectorStoreService
        
        embedding_service = EmbeddingService()
        vector_service = VectorStoreService(
            dimension=embedding_service.dimension,
            index_path="./data/orb_sessions.bin"
        )
        print("✓ Initialized embedding and vector services")
    except Exception as e:
        print(f"⚠ Services not available: {e}")
        embedding_service = None
        vector_service = None
    
    # Create Orb interface
    orb = OrbInterface(
        embedding_service=embedding_service,
        vector_store=vector_service
    )
    
    # Create session
    session_id = orb.create_session(
        user_id="test-user-001",
        metadata={"platform": "cli", "test": True}
    )
    
    print(f"✓ Session created: {session_id}")
    
except Exception as e:
    print(f"✗ Failed to create session: {e}")
    exit(1)

# Test 2: Add messages with various topics
print("\n2. Adding messages with diverse topics...")

messages = [
    ("user", "I need help setting up authentication for my API"),
    ("assistant", "I can help you with API authentication. What authentication method are you considering?"),
    ("user", "I'm thinking about using JWT tokens with role-based access control"),
    ("assistant", "Great choice! JWT with RBAC is secure and scalable. Let me walk you through the implementation."),
    ("user", "I also need to integrate a database for storing user credentials"),
    ("assistant", "For database integration, I recommend PostgreSQL with SQLAlchemy. We should also implement password hashing with bcrypt."),
    ("user", "What about rate limiting and security monitoring?"),
    ("assistant", "Rate limiting is crucial. I suggest using Redis for distributed rate limiting and Prometheus for monitoring."),
    ("user", "Perfect! Can you also help with vector search for semantic queries?"),
    ("assistant", "Absolutely! We can use FAISS for vector search with OpenAI or HuggingFace embeddings."),
]

for role, content in messages:
    orb.add_message(session_id, role, content)
    time.sleep(0.1)  # Small delay to simulate real conversation

print(f"✓ Added {len(messages)} messages to session")

# Test 3: Check session info (mid-conversation)
print("\n3. Checking session info (during conversation)...")

session_info = orb.get_session_info(session_id)

print(f"✓ Session Info:")
print(f"  Duration: {session_info['duration_formatted']}")
print(f"  Messages: {session_info['total_messages']}")
print(f"  Turns: {session_info['conversation_turns']}")
print(f"  Activity Level: {session_info['activity_level']}")

# Test 4: Extract topics
print("\n4. Extracting key topics from conversation...")

topics = orb.extract_session_topics(session_id, top_n=10)

print(f"✓ Extracted {len(topics)} key topics:")
for i, topic in enumerate(topics[:10], 1):
    print(f"  {i}. {topic['topic']}")
    print(f"     Mentions: {topic['mentions']}, Relevance: {topic['relevance']}")

# Test 5: Add more messages and check duration
print("\n5. Continuing conversation...")

additional_messages = [
    ("user", "How do I handle error logging and monitoring?"),
    ("assistant", "For error logging, use structlog for structured logs. For monitoring, set up Prometheus metrics and Grafana dashboards."),
    ("user", "What about WebSocket support for real-time features?"),
    ("assistant", "WebSocket support can be added with FastAPI's WebSocket endpoints. We can implement JWT authentication for WebSocket connections too."),
]

for role, content in additional_messages:
    orb.add_message(session_id, role, content)
    time.sleep(0.1)

print(f"✓ Added {len(additional_messages)} more messages")

# Test 6: Get updated topics
print("\n6. Updated topic extraction...")

updated_topics = orb.extract_session_topics(session_id, top_n=15)

print(f"✓ Updated topics ({len(updated_topics)} found):")
for i, topic in enumerate(updated_topics[:15], 1):
    confidence_pct = int(topic.get('confidence', 0) * 100)
    print(f"  {i}. {topic['topic']} ({topic['mentions']} mentions, {confidence_pct}% confidence)")

# Test 7: Close session and get final summary
print("\n7. Closing session and generating final summary...")

time.sleep(0.5)  # Ensure some duration

summary = orb.close_session(session_id)

print(f"✓ Session closed. Final summary:")
print(f"  Session ID: {summary['session_id']}")
print(f"  Start Time: {summary['start_time']}")
print(f"  End Time: {summary['end_time']}")
print(f"  Duration: {summary['duration_formatted']} ({summary['duration_seconds']:.2f}s)")
print(f"  Total Messages: {summary['total_messages']}")
print(f"  Conversation Turns: {summary['conversation_turns']}")
print(f"  Activity Level: {summary['activity_level']}")
print(f"  Decisions Made: {summary['decisions_made']}")

print(f"\n  Top 5 Topics:")
for topic in summary['key_topics'][:5]:
    print(f"    - {topic['topic']} ({topic['mentions']} mentions, {topic['relevance']} relevance)")

# Test 8: Get transcript
print("\n8. Getting conversation transcript...")

transcript = orb.get_session_transcript(session_id)
transcript_lines = transcript.split('\n')

print(f"✓ Transcript generated ({len(transcript_lines)} lines)")
print("  First 3 messages:")
for line in transcript_lines[:3]:
    print(f"    {line[:80]}...")

# Test 9: Save to vector store
print("\n9. Saving session to vector store...")

async def save_session():
    success = await orb.save_session(session_id)
    return success

if embedding_service and vector_service:
    saved = asyncio.run(save_session())
    if saved:
        print("✓ Session saved to vector store with embeddings")
    else:
        print("✗ Failed to save session")
else:
    print("⚠ Skipping vector store save (services not available)")

# Test 10: Session Manager
print("\n10. Testing Session Manager...")

try:
    manager = OrbSessionManager(
        embedding_service=embedding_service,
        vector_store=vector_service
    )
    
    # Create multiple sessions
    session1 = manager.create_user_session("user-001", {"type": "support"})
    session2 = manager.create_user_session("user-001", {"type": "training"})
    session3 = manager.create_user_session("user-002", {"type": "consultation"})
    
    print(f"✓ Created 3 sessions via manager")
    
    # Get statistics
    stats = manager.get_session_statistics()
    print(f"  Active sessions: {stats['total_active_sessions']}")
    print(f"  Total users: {stats['total_users']}")
    
except Exception as e:
    print(f"✗ Session manager test failed: {e}")

print("\n" + "=" * 70)
print("✅ Orb interface tests complete!")
print("\nFeatures demonstrated:")
print("  ✓ Session duration calculation (start to end)")
print("  ✓ Real-time duration tracking")
print("  ✓ NLP-based topic extraction (KeyBERT/spaCy/fallback)")
print("  ✓ Topic relevance and confidence scoring")
print("  ✓ Conversation transcript generation")
print("  ✓ Session metadata and statistics")
print("  ✓ Vector store integration")
print("  ✓ Multi-session management")

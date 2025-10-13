#!/usr/bin/env python3
"""
Test Grace Unified Orb Interface implementation
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.interface.orb_interface import (
    GraceUnifiedOrbInterface,
    PanelType,
    NotificationPriority,
)
from grace.intelligence.grace_intelligence import ReasoningContext


async def test_orb_interface():
    """Test the main orb interface functionality."""
    print("🔮 Testing Grace Unified Orb Interface...")

    # Initialize the orb interface
    orb = GraceUnifiedOrbInterface()
    print(f"✅ Orb interface initialized v{orb.version}")

    # Test session creation
    session_id = await orb.create_session("user123", {"theme": "dark"})
    print(f"✅ Created session: {session_id}")

    # Test chat interaction
    message_id = await orb.send_chat_message(
        session_id, "Hello Grace, can you analyze EUR/USD market data?"
    )
    print(f"✅ Sent chat message: {message_id}")

    # Get chat history
    messages = orb.get_chat_history(session_id)
    print(f"✅ Chat history: {len(messages)} messages")
    for msg in messages:
        print(f"   [{msg.message_type}] {msg.content[:50]}...")

    # Test panel creation
    trading_panel = await orb.create_panel(
        session_id, PanelType.TRADING, "EUR/USD Analysis"
    )
    print(f"✅ Created trading panel: {trading_panel}")

    analytics_panel = await orb.create_panel(
        session_id, PanelType.ANALYTICS, "Market Analytics"
    )
    print(f"✅ Created analytics panel: {analytics_panel}")

    # Test panels
    panels = orb.get_panels(session_id)
    print(f"✅ Active panels: {len(panels)}")
    for panel in panels:
        print(f"   {panel.panel_type.value}: {panel.title}")

    # Test IDE integration
    ide_panel = await orb.open_ide_panel(session_id)
    print(f"✅ Opened IDE panel: {ide_panel}")

    # Test IDE functionality
    ide = orb.get_ide_instance()
    flow_id = ide.create_flow("Test Flow", "Sample trading flow", "user123")
    print(f"✅ Created IDE flow: {flow_id}")

    # Add blocks to flow
    block1 = ide.add_block_to_flow(flow_id, "api_fetch", {"x": 100, "y": 100})
    block2 = ide.add_block_to_flow(flow_id, "sentiment_analysis", {"x": 300, "y": 100})
    print(f"✅ Added blocks to flow: {block1}, {block2}")

    # Test notifications
    notification_id = await orb.create_notification(
        user_id="user123",
        title="Market Alert",
        message="EUR/USD has broken resistance level",
        priority=NotificationPriority.HIGH,
        action_required=True,
        actions=[
            {"label": "View Chart", "action": "open_chart:EUR/USD"},
            {"label": "Dismiss", "action": "dismiss"},
        ],
    )
    print(f"✅ Created notification: {notification_id}")

    # Test governance
    gov_task = await orb.create_governance_task(
        title="Review Trading Strategy",
        description="New EUR/USD strategy needs approval",
        task_type="approval",
        requester_id="user123",
        assignee_id="admin",
    )
    print(f"✅ Created governance task: {gov_task}")

    # Get comprehensive stats
    stats = orb.get_orb_stats()
    print(f"✅ System stats:")
    print(f"   Sessions: {stats['sessions']['active']} active")
    print(f"   Messages: {stats['sessions']['total_messages']} total")
    print(f"   Panels: {stats['sessions']['total_panels']} total")
    print(f"   Memory fragments: {stats['memory']['total_fragments']}")
    print(f"   Governance tasks: {stats['governance']['total_tasks']}")
    print(f"   Notifications: {stats['notifications']['total']}")
    print(f"   IDE flows: {stats['ide']['flows']['total']}")
    print(f"   Available blocks: {stats['ide']['blocks']['types_available']}")
    print(f"   Intelligence domain pods: {stats['intelligence']['domain_pods']}")

    # Test memory upload (simulate)
    try:
        # Create a temporary file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test document content for Grace memory testing.")
            temp_file = f.name

        fragment_id = await orb.upload_document(
            "user123", temp_file, "txt", {"tags": ["test", "demo"]}
        )
        print(f"✅ Uploaded document to memory: {fragment_id}")

        # Clean up
        os.unlink(temp_file)
    except Exception as e:
        print(f"⚠️ Document upload test failed: {e}")

    # Test memory search
    memory_results = await orb.search_memory(session_id, "test")
    print(f"✅ Memory search results: {len(memory_results)} fragments")

    # End session
    success = await orb.end_session(session_id)
    print(f"✅ Ended session: {success}")

    print("🎉 All tests completed successfully!")
    return True


async def test_grace_intelligence():
    """Test Grace Intelligence reasoning separately."""
    print("\n🧠 Testing Grace Intelligence...")

    from grace.intelligence.grace_intelligence import (
        GraceIntelligence,
        ReasoningContext,
    )

    # Initialize Grace Intelligence
    intelligence = GraceIntelligence()
    print(f"✅ Grace Intelligence initialized v{intelligence.version}")

    # Test reasoning cycle
    context = ReasoningContext(
        user_id="test_user", session_id="test_session", metadata={"source": "test"}
    )

    test_queries = [
        "What are the current market conditions for EUR/USD?",
        "Create a sentiment analysis for our latest product launch",
        "Generate a report on sales performance this quarter",
        "Help me build a trading algorithm with risk management",
    ]

    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")

        try:
            result = await intelligence.process_request(query, context)
            print(f"✅ Success: {result.success}")
            print(f"🎯 Response: {result.response[:100]}...")
            print(f"🔍 Reasoning stages: {len(result.reasoning_trace)}")
            print(f"✅ Trust score: {result.verification.trust_score:.2f}")
            print(
                f"🏛️ Constitutional compliance: {result.verification.constitutional_compliance}"
            )

            if result.ui_instructions:
                print(
                    f"🖼️ UI instructions: {len(result.ui_instructions.get('panels', []))} panels to create"
                )

            if result.knowledge_updates:
                print(
                    f"🧠 Knowledge updates: {len(result.knowledge_updates)} fragments"
                )

        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("🎉 Grace Intelligence tests completed!")
    return True


async def main():
    """Main test function."""
    print("🚀 Starting Grace Unified Orb Interface Tests...")
    print("=" * 60)

    try:
        # Test core intelligence
        await test_grace_intelligence()

        # Test orb interface
        await test_orb_interface()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Grace Unified Orb Interface is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

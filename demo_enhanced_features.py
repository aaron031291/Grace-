#!/usr/bin/env python3
"""
Grace Enhanced Features Demo
Demonstrates the new functionality: knowledge base, task box, collaboration hub, and memory explorer.
"""

import asyncio
import json
from datetime import datetime
from grace.interface.orb_interface import GraceUnifiedOrbInterface, PanelType

async def demo_enhanced_features():
    """Demonstrate all the new enhanced features."""
    print("🚀 Grace Enhanced Features Demo")
    print("=" * 50)
    
    # Initialize Grace interface
    interface = GraceUnifiedOrbInterface()
    print(f"✅ Grace Interface v{interface.version} initialized")
    
    # Create a user session
    session_id = await interface.create_session("demo_user", {
        "theme": "professional",
        "language": "en"
    })
    print(f"✅ Session created: {session_id}")
    
    print("\n📚 KNOWLEDGE BASE & LIBRARY ACCESS DEMO")
    print("-" * 40)
    
    # Create some knowledge entries
    knowledge_entries = [
        {
            "title": "Python Pandas Library",
            "content": "Pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrame and Series for handling structured data efficiently.",
            "source": "documentation",
            "domain": "coding",
            "trust_score": 0.95,
            "relevance_tags": ["python", "data-analysis", "dataframe", "csv"],
            "related_libraries": ["numpy", "matplotlib", "seaborn"]
        },
        {
            "title": "Trading Algorithms Basics",
            "content": "Trading algorithms are computer programs that execute trades automatically based on predefined rules and market conditions. Key concepts include momentum, mean reversion, and risk management.",
            "source": "research_paper",
            "domain": "trading",
            "trust_score": 0.88,
            "relevance_tags": ["algorithms", "trading", "automation", "risk-management"],
            "related_libraries": ["pandas", "numpy", "ta-lib"]
        },
        {
            "title": "Machine Learning for Finance",
            "content": "Machine learning techniques can be applied to financial data for prediction, classification, and pattern recognition. Common approaches include regression, classification, and time series analysis.",
            "source": "academic",
            "domain": "analysis", 
            "trust_score": 0.92,
            "relevance_tags": ["machine-learning", "finance", "prediction", "time-series"],
            "related_libraries": ["scikit-learn", "tensorflow", "keras"]
        }
    ]
    
    knowledge_ids = []
    for entry in knowledge_entries:
        k_id = await interface.create_knowledge_entry(**entry)
        knowledge_ids.append(k_id)
        print(f"  📖 Created knowledge entry: {entry['title']} ({k_id})")
    
    # Test search functionality
    search_queries = ["pandas", "trading", "machine learning"]
    for query in search_queries:
        results = await interface.search_knowledge_base(query)
        print(f"  🔍 Search '{query}': {len(results)} results")
        for result in results[:2]:  # Show first 2 results
            print(f"    - {result.title} (trust: {result.trust_score}, accessed: {result.access_count})")
    
    # Test library data access
    library_data = await interface.access_library_data("pandas", "DataFrame operations")
    print(f"  📚 Library access: {library_data['library']} - {library_data['topic']}")
    
    # Open knowledge base panel
    kb_panel = await interface.open_knowledge_base_panel(session_id)
    print(f"  🎛️ Knowledge base panel opened: {kb_panel}")
    
    print("\n📋 TASK BOX DEMO")
    print("-" * 40)
    
    # Create various tasks
    tasks_data = [
        {
            "title": "Implement data cleaning pipeline",
            "description": "Create a robust data cleaning pipeline using pandas to preprocess trading data",
            "priority": "high",
            "assigned_to": "grace"
        },
        {
            "title": "Research momentum indicators", 
            "description": "Study various momentum indicators for algorithmic trading strategies",
            "priority": "medium",
            "assigned_to": "user"
        },
        {
            "title": "Backtest trading algorithm",
            "description": "Run historical backtests on the developed trading algorithm",
            "priority": "high", 
            "assigned_to": "grace"
        },
        {
            "title": "Document API endpoints",
            "description": "Create comprehensive documentation for all new API endpoints",
            "priority": "low",
            "assigned_to": "user"
        }
    ]
    
    task_ids = []
    for task in tasks_data:
        t_id = await interface.create_task_item(**task)
        task_ids.append(t_id)
        print(f"  ✅ Created task: {task['title']} ({t_id})")
    
    # Update some task statuses
    await interface.update_task_status(task_ids[0], "in_progress", 0.3)
    await interface.update_task_status(task_ids[2], "completed", 1.0)
    print(f"  🔄 Updated task statuses")
    
    # Merge relevant data into a task
    relevant_data = {
        "pandas_research": {
            "methods": ["read_csv", "dropna", "fillna", "groupby"],
            "performance_tips": "Use vectorized operations, avoid loops",
            "references": ["pandas documentation", "effective pandas book"]
        },
        "related_knowledge": knowledge_ids[0]  # Link to pandas knowledge entry
    }
    await interface.merge_task_data(task_ids[0], relevant_data)
    print(f"  🔗 Merged relevant data into task")
    
    # Show task summary
    all_tasks = interface.get_tasks_by_status()
    pending_tasks = interface.get_tasks_by_status("pending")
    completed_tasks = interface.get_tasks_by_status("completed")
    print(f"  📊 Task summary: {len(all_tasks)} total, {len(pending_tasks)} pending, {len(completed_tasks)} completed")
    
    # Open task box panel
    tb_panel = await interface.open_task_box_panel(session_id)
    print(f"  🎛️ Task box panel opened: {tb_panel}")
    
    print("\n🗂️ MEMORY EXPLORER DEMO")
    print("-" * 40)
    
    # Create memory structure (file-system like)
    projects_folder = await interface.create_memory_item("Projects", "folder")
    trading_folder = await interface.create_memory_item("Trading Algorithms", "folder", parent_id=projects_folder)
    analysis_folder = await interface.create_memory_item("Data Analysis", "folder", parent_id=projects_folder)
    print(f"  📁 Created folder structure")
    
    # Create some files
    strategy_file = await interface.create_memory_item(
        "momentum_strategy.py", 
        "file", 
        content="# Momentum-based trading strategy\nimport pandas as pd\nimport numpy as np\n\ndef calculate_momentum(prices, window=14):\n    return prices.pct_change(window)",
        parent_id=trading_folder
    )
    
    notes_file = await interface.create_memory_item(
        "research_notes.md",
        "file",
        content="# Research Notes\n\n## Key Findings\n- Momentum strategies work well in trending markets\n- Need to implement risk management\n- Backtesting shows 15% annual return",
        parent_id=analysis_folder
    )
    print(f"  📄 Created files with content")
    
    # Update file content
    updated_content = "# Updated Research Notes\n\n## Latest Analysis\n- Added stop-loss mechanism\n- Improved Sharpe ratio to 1.8\n- Ready for live testing"
    await interface.update_memory_item_content(notes_file, updated_content)
    print(f"  ✏️ Updated file content")
    
    # Show memory tree
    root_items = interface.get_memory_tree()
    print(f"  🌲 Memory tree: {len(root_items)} root items")
    for item in root_items:
        print(f"    - {item.name} ({item.item_type}) - {len(item.children)} children")
        if item.children:
            child_items = interface.get_memory_tree(item.item_id)
            for child in child_items[:3]:  # Show first 3 children
                print(f"      - {child.name} ({child.item_type})")
    
    # Open memory explorer panel
    me_panel = await interface.open_memory_explorer_panel(session_id)
    print(f"  🎛️ Memory explorer panel opened: {me_panel}")
    
    print("\n🤝 COLLABORATION HUB DEMO")
    print("-" * 40)
    
    # Create collaboration sessions
    collab_sessions = [
        {
            "topic": "Trading Algorithm Development",
            "participants": ["demo_user", "grace", "data_scientist"]
        },
        {
            "topic": "Market Data Analysis Pipeline",
            "participants": ["demo_user", "grace", "quantitative_analyst"]
        }
    ]
    
    collab_ids = []
    for session in collab_sessions:
        c_id = await interface.create_collaboration_session(**session)
        collab_ids.append(c_id)
        print(f"  💬 Created collaboration session: {session['topic']} ({c_id})")
    
    # Add discussion points
    discussion_points = [
        {
            "session_id": collab_ids[0],
            "author": "demo_user",
            "point": "We need to implement a robust backtesting framework before going live with the algorithm.",
            "point_type": "requirement"
        },
        {
            "session_id": collab_ids[0], 
            "author": "grace",
            "point": "I can help analyze the historical data and identify optimal parameters for the momentum strategy.",
            "point_type": "offer"
        },
        {
            "session_id": collab_ids[0],
            "author": "data_scientist",
            "point": "We should also consider market regime detection to adjust strategy parameters dynamically.",
            "point_type": "suggestion"
        }
    ]
    
    for point in discussion_points:
        await interface.add_discussion_point(**point)
        print(f"    💡 Added discussion point by {point['author']}")
    
    # Add action items
    action_items = [
        {
            "session_id": collab_ids[0],
            "title": "Implement backtesting framework",
            "description": "Create comprehensive backtesting system with risk metrics",
            "assigned_to": "grace",
            "priority": "high"
        },
        {
            "session_id": collab_ids[0],
            "title": "Market regime analysis",
            "description": "Research and implement market regime detection algorithms",
            "assigned_to": "data_scientist",
            "priority": "medium"
        }
    ]
    
    for action in action_items:
        await interface.add_action_item(**action)
        print(f"    ✅ Added action item: {action['title']} -> {action['assigned_to']}")
    
    # Open collaboration panel
    collab_panel = await interface.open_collaboration_panel(session_id, collab_ids[0])
    print(f"  🎛️ Collaboration panel opened: {collab_panel}")
    
    print("\n📊 COMPREHENSIVE STATISTICS")
    print("-" * 40)
    
    # Get comprehensive stats
    stats = interface.get_orb_stats()
    print(f"  Sessions: {stats['sessions']['active']} active")
    print(f"  Panels: {stats['sessions']['total_panels']} total panels open")
    print(f"  Enhanced Features:")
    for feature, count in stats['enhanced_features'].items():
        print(f"    - {feature.replace('_', ' ').title()}: {count}")
    
    print("\n🎯 SESSION PANEL SUMMARY")
    print("-" * 40)
    
    # Show all panels in the session
    session_panels = interface.get_panels(session_id)
    print(f"  Total panels in session: {len(session_panels)}")
    for panel in session_panels:
        print(f"    - {panel.title} ({panel.panel_type.value}) at ({panel.position['x']}, {panel.position['y']})")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"🚀 Grace now has enhanced capabilities for:")
    print(f"   • Knowledge base with library access")
    print(f"   • Task management with data merging")
    print(f"   • File-system-like memory exploration") 
    print(f"   • Collaborative development discussions")
    print(f"\n🌟 All features are accessible via both programmatic API and tab-based interface!")

if __name__ == "__main__":
    asyncio.run(demo_enhanced_features())
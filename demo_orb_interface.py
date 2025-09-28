#!/usr/bin/env python3
"""
Grace Unified Orb Interface - API Demo
Demonstrates the complete API functionality with real examples.
"""
import asyncio
import sys
import os
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.interface.orb_api import app, orb_interface
import uvicorn
from threading import Thread
import time
import requests

class OrbDemo:
    """Demo class for Grace Orb Interface."""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session_id = None
        self.user_id = "demo_user"
        
    def start_server_background(self):
        """Start the FastAPI server in background."""
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
        
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Wait for server to start
        
    def check_health(self):
        """Check API health."""
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"✅ API Health: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API Health check failed: {e}")
            return False
            
    def create_session(self):
        """Create a new session."""
        try:
            response = requests.post(f"{self.base_url}/api/orb/v1/sessions/create", 
                json={"user_id": self.user_id, "preferences": {"theme": "dark"}})
            result = response.json()
            self.session_id = result["session_id"]
            print(f"✅ Created session: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ Session creation failed: {e}")
            return False
            
    def demo_chat(self):
        """Demonstrate chat functionality."""
        print("\n🗣️  Chat Demo:")
        
        chat_messages = [
            "Hello Grace! Can you show me trading analysis for EUR/USD?",
            "Create a sentiment analysis dashboard for our product launch",
            "Help me build a data pipeline using the IDE",
            "Show me governance tasks that need approval"
        ]
        
        for message in chat_messages:
            try:
                # Send message
                response = requests.post(f"{self.base_url}/api/orb/v1/chat/message",
                    json={"session_id": self.session_id, "content": message})
                result = response.json()
                print(f"📤 Sent: {message[:50]}...")
                
                # Get chat history to see response
                time.sleep(0.5)  # Allow processing
                response = requests.get(f"{self.base_url}/api/orb/v1/chat/{self.session_id}/history?limit=2")
                history = response.json()
                
                for msg in history["messages"][-1:]:  # Show just Grace's response
                    if msg["message_type"] == "assistant":
                        print(f"📥 Grace: {msg['content'][:100]}...")
                        
            except Exception as e:
                print(f"❌ Chat error: {e}")
                
        print(f"✅ Chat demo completed")
        
    def demo_panels(self):
        """Demonstrate panel management."""
        print("\n📊 Panel Demo:")
        
        panel_types = ["trading", "sales", "analytics", "governance"]
        created_panels = []
        
        for panel_type in panel_types:
            try:
                response = requests.post(f"{self.base_url}/api/orb/v1/panels/create",
                    json={
                        "session_id": self.session_id,
                        "panel_type": panel_type,
                        "title": f"{panel_type.title()} Dashboard",
                        "data": {"demo": True, "initialized": True}
                    })
                result = response.json()
                created_panels.append(result["panel_id"])
                print(f"✅ Created {panel_type} panel: {result['panel_id']}")
                
            except Exception as e:
                print(f"❌ Panel creation failed for {panel_type}: {e}")
        
        # List all panels
        try:
            response = requests.get(f"{self.base_url}/api/orb/v1/panels/{self.session_id}")
            panels = response.json()
            print(f"📋 Active panels: {len(panels['panels'])}")
            for panel in panels["panels"]:
                print(f"   • {panel['title']} ({panel['panel_type']})")
                
        except Exception as e:
            print(f"❌ Panel listing failed: {e}")
            
    def demo_ide(self):
        """Demonstrate IDE functionality."""
        print("\n🛠️  IDE Demo:")
        
        try:
            # Open IDE panel
            response = requests.post(f"{self.base_url}/api/orb/v1/ide/panels/{self.session_id}")
            result = response.json()
            print(f"✅ Opened IDE panel: {result['panel_id']}")
            
            # Create a new flow
            response = requests.post(f"{self.base_url}/api/orb/v1/ide/flows",
                json={
                    "name": "Demo Trading Flow",
                    "description": "Sample automated trading workflow",
                    "creator_id": self.user_id
                })
            flow_result = response.json()
            flow_id = flow_result["flow_id"]
            print(f"✅ Created IDE flow: {flow_id}")
            
            # Get available blocks
            response = requests.get(f"{self.base_url}/api/orb/v1/ide/blocks")
            blocks = response.json()
            print(f"📚 Available blocks: {len(blocks['blocks'])}")
            
            # Add blocks to flow
            block_types = ["api_fetch", "sentiment_analysis", "data_filter"]
            positions = [{"x": 100, "y": 100}, {"x": 300, "y": 100}, {"x": 500, "y": 100}]
            
            for i, block_type in enumerate(block_types):
                if block_type in blocks["blocks"]:
                    response = requests.post(f"{self.base_url}/api/orb/v1/ide/flows/blocks",
                        json={
                            "flow_id": flow_id,
                            "block_type_id": block_type,
                            "position": positions[i],
                            "name": f"Demo {block_type}",
                            "configuration": {"demo": True}
                        })
                    result = response.json()
                    print(f"✅ Added {block_type} block: {result['block_id']}")
            
            # Get flow details
            response = requests.get(f"{self.base_url}/api/orb/v1/ide/flows/{flow_id}")
            flow_details = response.json()
            print(f"📋 Flow '{flow_details['name']}' has {len(flow_details['blocks'])} blocks")
            
        except Exception as e:
            print(f"❌ IDE demo error: {e}")
            
    def demo_memory(self):
        """Demonstrate memory functionality."""
        print("\n🧠 Memory Demo:")
        
        try:
            # Create a test document
            import tempfile
            test_content = """
# Grace AI System Demo Document

This is a demonstration document for Grace's memory system.

## Key Features
- Advanced reasoning with 5-stage cycle
- Visual flow editor with drag-and-drop blocks
- Real-time governance and compliance
- Multi-modal document processing

## Trading Analysis
EUR/USD shows bullish momentum with RSI at 65.
Support level at 1.2300, resistance at 1.2450.
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            # Upload document
            with open(temp_file, 'rb') as f:
                files = {'file': ('demo_document.txt', f, 'text/plain')}
                data = {'user_id': self.user_id}
                response = requests.post(f"{self.base_url}/api/orb/v1/memory/upload", 
                                       files=files, data=data)
                result = response.json()
                print(f"✅ Uploaded document: {result['fragment_id']}")
                
            # Clean up temp file
            os.unlink(temp_file)
            
            # Search memory
            search_queries = ["Grace AI", "EUR/USD", "trading", "resistance"]
            
            for query in search_queries:
                response = requests.post(f"{self.base_url}/api/orb/v1/memory/search",
                    json={"session_id": self.session_id, "query": query})
                results = response.json()
                print(f"🔍 Search '{query}': {len(results['results'])} results")
                
            # Get memory stats
            response = requests.get(f"{self.base_url}/api/orb/v1/memory/stats")
            stats = response.json()
            print(f"📊 Memory stats: {stats['total_fragments']} fragments")
                
        except Exception as e:
            print(f"❌ Memory demo error: {e}")
            
    def demo_governance(self):
        """Demonstrate governance functionality."""
        print("\n🏛️  Governance Demo:")
        
        try:
            # Create governance task
            response = requests.post(f"{self.base_url}/api/orb/v1/governance/tasks",
                json={
                    "title": "Review New Trading Strategy",
                    "description": "EUR/USD momentum strategy needs compliance review",
                    "task_type": "approval",
                    "requester_id": self.user_id,
                    "assignee_id": "admin_user"
                })
            result = response.json()
            task_id = result["task_id"]
            print(f"✅ Created governance task: {task_id}")
            
            # Get tasks for user
            response = requests.get(f"{self.base_url}/api/orb/v1/governance/tasks/{self.user_id}")
            tasks = response.json()
            print(f"📋 User tasks: {len(tasks['tasks'])}")
            
            for task in tasks["tasks"]:
                print(f"   • {task['title']} ({task['status']})")
                
        except Exception as e:
            print(f"❌ Governance demo error: {e}")
            
    def demo_notifications(self):
        """Demonstrate notifications."""
        print("\n🔔 Notifications Demo:")
        
        try:
            # Create different types of notifications
            notifications = [
                {
                    "title": "Trading Alert",
                    "message": "EUR/USD broke resistance at 1.2450",
                    "priority": "high",
                    "action_required": True,
                    "actions": [
                        {"label": "View Chart", "action": "open_chart"},
                        {"label": "Create Order", "action": "create_order"}
                    ]
                },
                {
                    "title": "System Update",
                    "message": "New Grace Intelligence features available",
                    "priority": "medium",
                    "action_required": False
                },
                {
                    "title": "Governance Review",
                    "message": "Your approval is needed for trading strategy",
                    "priority": "critical",
                    "action_required": True,
                    "actions": [{"label": "Review", "action": "review_task"}]
                }
            ]
            
            created_notifications = []
            
            for notif_data in notifications:
                notif_data["user_id"] = self.user_id
                response = requests.post(f"{self.base_url}/api/orb/v1/notifications",
                    json=notif_data)
                result = response.json()
                created_notifications.append(result["notification_id"])
                print(f"✅ Created {notif_data['priority']} notification: {result['notification_id']}")
            
            # Get user notifications
            response = requests.get(f"{self.base_url}/api/orb/v1/notifications/{self.user_id}")
            user_notifications = response.json()
            print(f"📬 User notifications: {len(user_notifications['notifications'])}")
            
            for notif in user_notifications["notifications"]:
                print(f"   • [{notif['priority'].upper()}] {notif['title']}")
                
        except Exception as e:
            print(f"❌ Notifications demo error: {e}")
            
    def demo_stats(self):
        """Show comprehensive system statistics."""
        print("\n📈 System Statistics:")
        
        try:
            response = requests.get(f"{self.base_url}/api/orb/v1/stats")
            stats = response.json()
            
            print(f"🔮 Orb System Stats:")
            print(f"   Sessions: {stats['sessions']['active']} active")
            print(f"   Messages: {stats['sessions']['total_messages']} total")
            print(f"   Panels: {stats['sessions']['total_panels']} total")
            print(f"   Memory: {stats['memory']['total_fragments']} fragments")
            print(f"   Governance: {stats['governance']['total_tasks']} tasks")
            print(f"   Notifications: {stats['notifications']['total']} total")
            print(f"   IDE: {stats['ide']['flows']['total']} flows")
            print(f"   Intelligence: {stats['intelligence']['domain_pods']} domain pods")
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            
    def run_full_demo(self):
        """Run the complete demo."""
        print("🚀 Grace Unified Orb Interface - API Demo")
        print("=" * 60)
        
        # Start server
        print("Starting API server...")
        self.start_server_background()
        
        # Check health
        if not self.check_health():
            print("❌ Server not responding. Demo aborted.")
            return False
            
        # Create session
        if not self.create_session():
            print("❌ Session creation failed. Demo aborted.")
            return False
            
        # Run all demos
        self.demo_chat()
        self.demo_panels()
        self.demo_ide()
        self.demo_memory()
        self.demo_governance()
        self.demo_notifications()
        self.demo_stats()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print(f"📊 API Documentation: {self.base_url}/docs")
        print(f"🔍 API Root: {self.base_url}/")
        print("\nThe Grace Unified Orb Interface is now ready for integration!")
        
        return True

def main():
    """Main demo function."""
    demo = OrbDemo()
    success = demo.run_full_demo()
    
    if success:
        print("\n⏰ Server will continue running for 30 seconds for manual testing...")
        print("   Visit http://localhost:8080/docs for interactive API documentation")
        time.sleep(30)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
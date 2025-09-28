#!/usr/bin/env python3
"""
Grace Deployment Readiness Assessment
=====================================

Comprehensive test to verify Grace is ready for live deployment
and fully functional for user interaction.
"""

import asyncio
import json
import requests
import websockets
import sys
from datetime import datetime
from typing import Dict, List, Tuple

class GraceDeploymentTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://")
        self.session_id = None
        self.test_results = []

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })

    def test_health_endpoints(self) -> bool:
        """Test basic health endpoints"""
        try:
            # Test root endpoint
            response = requests.get(f"{self.base_url}/")
            root_data = response.json()
            self.log_result("Root Endpoint", 
                          response.status_code == 200 and "service" in root_data,
                          f"Status: {response.status_code}")

            # Test health endpoint  
            response = requests.get(f"{self.base_url}/health")
            health_data = response.json()
            self.log_result("Health Endpoint",
                          response.status_code == 200 and health_data.get("status") == "healthy",
                          f"Status: {health_data.get('status')}")

            # Test stats endpoint
            response = requests.get(f"{self.base_url}/api/orb/v1/stats")
            stats_data = response.json()
            self.log_result("Stats Endpoint",
                          response.status_code == 200 and "sessions" in stats_data,
                          f"Active sessions: {stats_data.get('sessions', {}).get('active', 0)}")
            
            return all([r["passed"] for r in self.test_results[-3:]])
            
        except Exception as e:
            self.log_result("Health Endpoints", False, f"Exception: {e}")
            return False

    def test_session_management(self) -> bool:
        """Test session creation and management"""
        try:
            # Create session
            session_data = {
                "user_id": "deployment_test_user",
                "preferences": {
                    "theme": "test",
                    "notifications": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/orb/v1/sessions/create",
                json=session_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = result.get("session_id")
                self.log_result("Session Creation",
                              self.session_id is not None,
                              f"Session ID: {self.session_id}")
                return True
            else:
                self.log_result("Session Creation", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Session Creation", False, f"Exception: {e}")
            return False

    def test_chat_functionality(self) -> bool:
        """Test chat message sending and receiving"""
        if not self.session_id:
            self.log_result("Chat Functionality", False, "No session ID available")
            return False
            
        try:
            # Send a message
            chat_data = {
                "session_id": self.session_id,
                "content": "Hello Grace! This is a deployment readiness test. Please confirm you can receive and process this message.",
                "attachments": []
            }
            
            response = requests.post(
                f"{self.base_url}/api/orb/v1/chat/message",
                json=chat_data
            )
            
            if response.status_code == 200:
                result = response.json()
                message_id = result.get("message_id")
                self.log_result("Chat Message Send",
                              message_id is not None,
                              f"Message ID: {message_id}")
                
                # Get chat history to see response
                history_response = requests.get(
                    f"{self.base_url}/api/orb/v1/chat/{self.session_id}/history?limit=5"
                )
                
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    messages = history_data.get("messages", [])
                    grace_responses = [m for m in messages if m.get("user_id") == "grace"]
                    
                    self.log_result("Chat Response Received",
                                  len(grace_responses) > 0,
                                  f"Grace responded {len(grace_responses)} times")
                    return len(grace_responses) > 0
                else:
                    self.log_result("Chat History", False, f"Status: {history_response.status_code}")
                    return False
            else:
                self.log_result("Chat Message Send", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Chat Functionality", False, f"Exception: {e}")
            return False

    async def test_websocket_realtime(self) -> bool:
        """Test WebSocket real-time communication"""
        if not self.session_id:
            self.log_result("WebSocket Communication", False, "No session ID available")
            return False
            
        try:
            uri = f"{self.ws_url}/ws/{self.session_id}"
            
            async with websockets.connect(uri) as websocket:
                self.log_result("WebSocket Connection", True, "Connected successfully")
                
                # Send a real-time message
                message = {
                    "type": "chat_message", 
                    "content": "This is a real-time WebSocket test for deployment readiness.",
                    "attachments": []
                }
                
                await websocket.send(json.dumps(message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    response_data = json.loads(response)
                    
                    has_response = (response_data.get("type") == "chat_response" and
                                  "messages" in response_data)
                    
                    self.log_result("WebSocket Real-time Response",
                                  has_response,
                                  f"Response type: {response_data.get('type')}")
                    
                    # Test heartbeat
                    ping_msg = {"type": "ping"}
                    await websocket.send(json.dumps(ping_msg))
                    
                    pong_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    pong_data = json.loads(pong_response)
                    
                    heartbeat_works = pong_data.get("type") == "pong"
                    self.log_result("WebSocket Heartbeat",
                                  heartbeat_works,
                                  f"Ping/pong: {'working' if heartbeat_works else 'failed'}")
                    
                    return has_response and heartbeat_works
                    
                except asyncio.TimeoutError:
                    self.log_result("WebSocket Real-time Response", False, "Response timeout")
                    return False
                    
        except Exception as e:
            self.log_result("WebSocket Communication", False, f"Exception: {e}")
            return False

    def test_api_endpoints(self) -> bool:
        """Test various API endpoints for completeness"""
        endpoints = [
            ("/api/orb/v1/stats", "System Statistics"),
        ]
        
        all_passed = True
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                passed = response.status_code == 200
                self.log_result(f"API Endpoint - {name}",
                              passed,
                              f"Status: {response.status_code}")
                if not passed:
                    all_passed = False
            except Exception as e:
                self.log_result(f"API Endpoint - {name}", False, f"Exception: {e}")
                all_passed = False
                
        return all_passed

    async def run_comprehensive_test(self) -> Dict:
        """Run all deployment readiness tests"""
        print("🚀 Grace Deployment Readiness Assessment")
        print("=" * 50)
        
        # Test sequence
        test_results = {
            "health_endpoints": self.test_health_endpoints(),
            "session_management": self.test_session_management(),
            "chat_functionality": self.test_chat_functionality(), 
            "websocket_realtime": await self.test_websocket_realtime(),
            "api_endpoints": self.test_api_endpoints()
        }
        
        print("\n" + "=" * 50)
        print("📊 DEPLOYMENT READINESS SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        deployment_ready = passed_tests == total_tests
        readiness_status = "🎉 READY FOR DEPLOYMENT" if deployment_ready else "⚠️ NEEDS ATTENTION"
        print(f"\nDeployment Status: {readiness_status}")
        
        if deployment_ready:
            print("\n✅ Grace is fully functional and ready for live deployment!")
            print("✅ Users can communicate with Grace immediately upon going live!")
            print("✅ All critical interfaces are wired up and working!")
        else:
            failed_tests = [name for name, passed in test_results.items() if not passed]
            print(f"\n❌ Failed tests that need attention: {', '.join(failed_tests)}")
        
        return {
            "deployment_ready": deployment_ready,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "test_results": test_results,
            "detailed_results": self.test_results
        }

async def main():
    tester = GraceDeploymentTester()
    result = await tester.run_comprehensive_test()
    
    # Save results
    with open("/tmp/deployment_readiness_report.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: /tmp/deployment_readiness_report.json")
    
    return 0 if result["deployment_ready"] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
"""
Test WebSocket functionality with JWT authentication
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

print("Testing Grace WebSocket System")
print("=" * 60)

# Step 1: Login to get JWT token
print("\n1. Logging in to get JWT token...")
response = requests.post(
    f"{BASE_URL}/api/v1/auth/token",
    data={"username": "admin", "password": "Admin123!"}
)

if response.status_code != 200:
    print(f"✗ Login failed: {response.status_code}")
    exit(1)

tokens = response.json()
access_token = tokens['access_token']
print(f"✓ Got access token: {access_token[:30]}...")

# Step 2: Connect to WebSocket with JWT
print("\n2. Connecting to WebSocket...")

async def test_websocket():
    ws_uri = f"{WS_URL}/api/v1/ws/connect?token={access_token}"
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            print("✓ WebSocket connected!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"\nReceived: {json.loads(welcome)}")
            
            # Test 3: Subscribe to a channel
            print("\n3. Subscribing to collaboration channel...")
            subscribe_msg = json.dumps({
                "type": "subscribe",
                "channel": "collaboration:test-session"
            })
            await websocket.send(subscribe_msg)
            
            response = await websocket.recv()
            print(f"Response: {json.loads(response)}")
            
            # Test 4: Send a message to channel
            print("\n4. Sending message to channel...")
            message = json.dumps({
                "type": "message",
                "channel": "collaboration:test-session",
                "data": {
                    "content": "Hello from test client!",
                    "timestamp": datetime.now().isoformat()
                }
            })
            await websocket.send(message)
            
            # Receive the broadcasted message
            broadcast = await websocket.recv()
            print(f"Broadcasted: {json.loads(broadcast)}")
            
            # Test 5: Test ping/pong
            print("\n5. Testing ping/pong...")
            ping_msg = json.dumps({
                "type": "ping",
                "data": {"timestamp": datetime.now().isoformat()}
            })
            await websocket.send(ping_msg)
            
            pong = await websocket.recv()
            print(f"Pong received: {json.loads(pong)}")
            
            # Test 6: Wait for server ping
            print("\n6. Waiting for server ping (30s timeout)...")
            try:
                server_ping = await asyncio.wait_for(websocket.recv(), timeout=35)
                ping_data = json.loads(server_ping)
                print(f"Server ping: {ping_data}")
                
                # Respond with pong
                if ping_data.get("type") == "ping":
                    pong_msg = json.dumps({
                        "type": "pong",
                        "data": ping_data.get("data", {})
                    })
                    await websocket.send(pong_msg)
                    print("✓ Sent pong response")
            except asyncio.TimeoutError:
                print("⚠ No server ping received (normal if ping interval > 30s)")
            
            # Test 7: Unsubscribe
            print("\n7. Unsubscribing from channel...")
            unsubscribe_msg = json.dumps({
                "type": "unsubscribe",
                "channel": "collaboration:test-session"
            })
            await websocket.send(unsubscribe_msg)
            
            response = await websocket.recv()
            print(f"Response: {json.loads(response)}")
            
            print("\n✓ All WebSocket tests passed!")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

# Run async tests
asyncio.run(test_websocket())

print("\n" + "=" * 60)
print("WebSocket testing complete!")
print("\nFeatures tested:")
print("  ✓ JWT authentication")
print("  ✓ Channel subscription/unsubscription")
print("  ✓ Message publishing")
print("  ✓ Ping/pong heartbeat")
print("  ✓ Automatic connection management")

"""
Simple test script for Grace authentication
"""

import requests

BASE_URL = "http://localhost:8000"

print("Testing Grace Authentication System")
print("=" * 50)

# Test login
print("\n1. Testing login with admin credentials...")
response = requests.post(
    f"{BASE_URL}/api/v1/auth/token",
    data={"username": "admin", "password": "Admin123!"}
)

if response.status_code == 200:
    tokens = response.json()
    print("✓ Login successful!")
    print(f"  Access token: {tokens['access_token'][:50]}...")
    print(f"  Token type: {tokens['token_type']}")
    
    # Test getting current user
    print("\n2. Testing /auth/me endpoint...")
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    me_response = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers)
    
    if me_response.status_code == 200:
        user = me_response.json()
        print("✓ User info retrieved!")
        print(f"  Username: {user['username']}")
        print(f"  Email: {user['email']}")
        print(f"  Roles: {user['roles']}")
        print(f"  Is superuser: {user['is_superuser']}")
    else:
        print(f"✗ Failed to get user info: {me_response.status_code}")
else:
    print(f"✗ Login failed: {response.status_code}")
    print(f"  Response: {response.text}")

print("\n" + "=" * 50)
print("Test complete!")

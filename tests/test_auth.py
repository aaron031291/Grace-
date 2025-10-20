"""
Test script for Grace authentication system
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_login():
    """Test login with default admin credentials"""
    print("\n=== Testing Login ===")
    
    # Try login
    login_data = {
        "username": "admin",
        "password": "Admin123!"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/token",
        data=login_data
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        tokens = response.json()
        print(f"Access Token: {tokens['access_token'][:50]}...")
        print(f"Refresh Token: {tokens['refresh_token'][:50]}...")
        print(f"Token Type: {tokens['token_type']}")
        print(f"Expires In: {tokens['expires_in']} seconds")
        return tokens
    else:
        print(f"Error: {response.json()}")
        return None


def test_get_current_user(access_token):
    """Test getting current user info"""
    print("\n=== Testing Get Current User ===")
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(
        f"{BASE_URL}/api/v1/auth/me",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        user = response.json()
        print(f"User ID: {user['id']}")
        print(f"Username: {user['username']}")
        print(f"Email: {user['email']}")
        print(f"Roles: {user['roles']}")
        print(f"Is Superuser: {user['is_superuser']}")
        print(f"Is Active: {user['is_active']}")
        print(f"Is Verified: {user['is_verified']}")
        return user
    else:
        print(f"Error: {response.json()}")
        return None


def test_register_user():
    """Test user registration"""
    print("\n=== Testing User Registration ===")
    
    user_data = {
        "username": f"testuser_{datetime.now().timestamp()}",
        "email": f"test_{datetime.now().timestamp()}@example.com",
        "password": "TestPass123!",
        "full_name": "Test User"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/register",
        json=user_data
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        user = response.json()
        print(f"Created User: {user['username']}")
        print(f"User ID: {user['id']}")
        print(f"Email: {user['email']}")
        print(f"Roles: {user['roles']}")
        return user
    else:
        print(f"Error: {response.json()}")
        return None


def test_refresh_token(refresh_token):
    """Test token refresh"""
    print("\n=== Testing Token Refresh ===")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        tokens = response.json()
        print(f"New Access Token: {tokens['access_token'][:50]}...")
        print(f"New Refresh Token: {tokens['refresh_token'][:50]}...")
        return tokens
    else:
        print(f"Error: {response.json()}")
        return None


def test_invalid_login():
    """Test login with invalid credentials"""
    print("\n=== Testing Invalid Login ===")
    
    login_data = {
        "username": "admin",
        "password": "WrongPassword!"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/token",
        data=login_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Expected 401 Unauthorized")
    
    if response.status_code == 401:
        print("✓ Correctly rejected invalid credentials")
        return True
    else:
        print("✗ Should have rejected invalid credentials")
        return False


def test_unauthorized_access():
    """Test accessing protected endpoint without token"""
    print("\n=== Testing Unauthorized Access ===")
    
    response = requests.get(f"{BASE_URL}/api/v1/auth/me")
    
    print(f"Status: {response.status_code}")
    print(f"Expected 401 Unauthorized")
    
    if response.status_code == 401:
        print("✓ Correctly blocked unauthorized access")
        return True
    else:
        print("✗ Should have blocked unauthorized access")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Grace Authentication System - Test Suite")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ API is not healthy. Make sure the server is running.")
        return
    
    # Test unauthorized access
    test_unauthorized_access()
    
    # Test invalid login
    test_invalid_login()
    
    # Test login
    tokens = test_login()
    if not tokens:
        print("\n❌ Login failed. Tests cannot continue.")
        return
    
    access_token = tokens['access_token']
    refresh_token = tokens['refresh_token']
    
    # Test get current user
    user = test_get_current_user(access_token)
    
    # Test user registration
    new_user = test_register_user()
    
    # Test token refresh
    new_tokens = test_refresh_token(refresh_token)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nDefault Admin Credentials:")
    print("  Username: admin")
    print("  Password: Admin123!")
    print("\nAPI Documentation: http://localhost:8000/api/docs")


if __name__ == "__main__":
    main()

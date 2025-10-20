"""
Test script for policies, sessions, and tasks
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

print("Testing Grace Governance System")
print("=" * 60)

# Step 1: Login
print("\n1. Logging in...")
response = requests.post(
    f"{BASE_URL}/api/v1/auth/token",
    data={"username": "admin", "password": "Admin123!"}
)

if response.status_code != 200:
    print(f"✗ Login failed: {response.status_code}")
    exit(1)

tokens = response.json()
headers = {"Authorization": f"Bearer {tokens['access_token']}"}
print("✓ Login successful")

# Step 2: Create policies
print("\n2. Creating policies...")
policy_data = {
    "name": "Data Privacy Policy",
    "description": "Ensures proper handling of user data and privacy compliance",
    "policy_type": "PRIVACY",
    "rules": [
        {"type": "encryption", "required": True},
        {"type": "consent", "required": True}
    ],
    "constraints": {"data_retention_days": 90},
    "metadata": {"department": "security"}
}

response = requests.post(
    f"{BASE_URL}/api/v1/policies",
    headers=headers,
    json=policy_data
)

if response.status_code == 201:
    policy = response.json()
    policy_id = policy['id']
    print(f"✓ Policy created: {policy['name']} (ID: {policy_id})")
else:
    print(f"✗ Failed to create policy: {response.status_code}")
    policy_id = None

# Step 3: List policies
print("\n3. Listing policies...")
response = requests.get(f"{BASE_URL}/api/v1/policies", headers=headers)
if response.status_code == 200:
    policies = response.json()
    print(f"✓ Found {len(policies)} policies")
    for p in policies:
        print(f"  - {p['name']} ({p['policy_type']}) - {p['status']}")

# Step 4: Create collaboration session
print("\n4. Creating collaboration session...")
session_data = {
    "name": "AI Ethics Review Session",
    "description": "Review ethical implications of new AI features",
    "session_type": "ethics_review",
    "context": {"topic": "AI fairness", "priority": "high"}
}

response = requests.post(
    f"{BASE_URL}/api/v1/sessions",
    headers=headers,
    json=session_data
)

if response.status_code == 201:
    session = response.json()
    session_id = session['id']
    print(f"✓ Session created: {session['name']} (ID: {session_id})")
else:
    print(f"✗ Failed to create session: {response.status_code}")
    session_id = None

# Step 5: Add message to session
if session_id:
    print("\n5. Adding message to session...")
    message_data = {
        "content": "Let's discuss the fairness metrics for our recommendation system",
        "message_type": "text"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/sessions/{session_id}/messages",
        headers=headers,
        json=message_data
    )
    
    if response.status_code == 200:
        print("✓ Message added to session")

# Step 6: Create tasks
print("\n6. Creating tasks...")
tasks_data = [
    {
        "title": "Review fairness metrics",
        "description": "Analyze bias in recommendation algorithm",
        "priority": "HIGH",
        "session_id": session_id,
        "tags": ["ethics", "fairness", "review"],
        "estimated_hours": 4.0,
        "due_date": (datetime.now() + timedelta(days=3)).isoformat()
    },
    {
        "title": "Update privacy policy",
        "description": "Incorporate new data handling procedures",
        "priority": "MEDIUM",
        "policy_id": policy_id,
        "tags": ["policy", "privacy"],
        "estimated_hours": 2.0,
        "due_date": (datetime.now() + timedelta(days=7)).isoformat()
    }
]

task_ids = []
for task_data in tasks_data:
    response = requests.post(
        f"{BASE_URL}/api/v1/tasks",
        headers=headers,
        json=task_data
    )
    
    if response.status_code == 201:
        task = response.json()
        task_ids.append(task['id'])
        print(f"✓ Task created: {task['title']} ({task['priority']})")

# Step 7: List tasks
print("\n7. Listing tasks...")
response = requests.get(f"{BASE_URL}/api/v1/tasks", headers=headers)
if response.status_code == 200:
    tasks = response.json()
    print(f"✓ Found {len(tasks)} tasks")
    for t in tasks:
        print(f"  - {t['title']} [{t['status']}] - Priority: {t['priority']}")

# Step 8: Update task status
if task_ids:
    print(f"\n8. Updating task status...")
    response = requests.put(
        f"{BASE_URL}/api/v1/tasks/{task_ids[0]}",
        headers=headers,
        json={"status": "IN_PROGRESS", "progress_percentage": 25}
    )
    
    if response.status_code == 200:
        print("✓ Task status updated to IN_PROGRESS")

# Step 9: Update session status
if session_id:
    print(f"\n9. Updating session status...")
    response = requests.put(
        f"{BASE_URL}/api/v1/sessions/{session_id}",
        headers=headers,
        json={"status": "PAUSED"}
    )
    
    if response.status_code == 200:
        print("✓ Session status updated to PAUSED")

# Step 10: Get policy details
if policy_id:
    print(f"\n10. Getting policy details...")
    response = requests.get(f"{BASE_URL}/api/v1/policies/{policy_id}", headers=headers)
    if response.status_code == 200:
        policy = response.json()
        print(f"✓ Policy: {policy['name']}")
        print(f"  Type: {policy['policy_type']}")
        print(f"  Status: {policy['status']}")
        print(f"  Rules: {len(policy.get('rules', []))}")

print("\n" + "=" * 60)
print("✅ Governance tests complete!")
print("\nAPI Documentation: http://localhost:8000/api/docs")
print("\nEndpoints:")
print("  - Policies: /api/v1/policies")
print("  - Sessions: /api/v1/sessions")
print("  - Tasks: /api/v1/tasks")

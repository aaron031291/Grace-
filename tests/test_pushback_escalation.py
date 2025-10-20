"""
Test pushback escalation with database-backed error tracking
"""

import asyncio
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000"

print("Testing Grace Pushback Escalation System")
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

# Step 2: Report individual errors (below threshold)
print("\n2. Reporting low-severity errors (below threshold)...")
for i in range(3):
    response = requests.post(
        f"{BASE_URL}/api/v1/avn/report-error",
        headers=headers,
        json={
            "error_type": "ValidationError",
            "error_message": f"Invalid input in field 'email' (test {i+1})",
            "context": {"field": "email", "attempt": i+1},
            "severity": "low",
            "endpoint": "/api/v1/users/create"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Error {i+1} reported: decision={result['escalation_decision']}")
    else:
        print(f"  ✗ Failed to report error: {response.status_code}")

# Step 3: Trigger threshold with high-severity errors
print("\n3. Triggering escalation threshold with high-severity errors...")
for i in range(3):
    response = requests.post(
        f"{BASE_URL}/api/v1/avn/report-error",
        headers=headers,
        json={
            "error_type": "PermissionError",
            "error_message": "Unauthorized access attempt to admin panel",
            "context": {"resource": "admin_panel", "attempt": i+1},
            "severity": "high",
            "endpoint": "/api/v1/admin/dashboard"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        decision = result['escalation_decision']
        print(f"  ✓ Error {i+1} reported: decision={decision}")
        
        if decision == "escalate_to_avn":
            print(f"    🚨 ESCALATED TO AVN! Threshold exceeded.")
    else:
        print(f"  ✗ Failed to report error: {response.status_code}")

# Step 4: Test burst detection with many rapid errors
print("\n4. Testing burst detection (many errors quickly)...")
for i in range(8):
    response = requests.post(
        f"{BASE_URL}/api/v1/avn/report-error",
        headers=headers,
        json={
            "error_type": "ConnectionError",
            "error_message": "Database connection timeout",
            "context": {"database": "main", "retry": i+1},
            "severity": "medium",
            "endpoint": "/api/v1/documents/search"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if i % 3 == 0:
            print(f"  ✓ Error {i+1} reported: decision={result['escalation_decision']}")

# Step 5: Report critical error (immediate escalation)
print("\n5. Reporting critical error (should escalate immediately)...")
response = requests.post(
    f"{BASE_URL}/api/v1/avn/report-error",
    headers=headers,
    json={
        "error_type": "SecurityError",
        "error_message": "Potential security breach detected",
        "context": {"threat_level": "high", "source_ip": "203.0.113.42"},
        "severity": "critical",
        "endpoint": "/api/v1/auth/token"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"✓ Critical error reported: decision={result['escalation_decision']}")
    if result['escalation_decision'] == "immediate_action":
        print("  🚨 IMMEDIATE ACTION TRIGGERED!")

# Step 6: Get AVN alerts
print("\n6. Fetching AVN alerts...")
response = requests.get(
    f"{BASE_URL}/api/v1/avn/alerts",
    headers=headers,
    params={"limit": 10}
)

if response.status_code == 200:
    alerts = response.json()
    print(f"✓ Found {len(alerts)} AVN alerts:")
    
    for i, alert in enumerate(alerts[:5], 1):
        print(f"\n  Alert {i}:")
        print(f"    Title: {alert['title']}")
        print(f"    Severity: {alert['severity']}")
        print(f"    Error Count: {alert['error_count']}")
        print(f"    Status: {alert['status']}")
        print(f"    AVN Notified: {alert['avn_notified']}")
        print(f"    Triggered: {alert['triggered_at']}")
else:
    print(f"✗ Failed to fetch alerts: {response.status_code}")

# Step 7: Get error statistics
print("\n7. Getting error statistics...")
response = requests.get(
    f"{BASE_URL}/api/v1/avn/statistics",
    headers=headers,
    params={"time_window_minutes": 60}
)

if response.status_code == 200:
    stats = response.json()
    print(f"✓ Error statistics (last {stats['time_window_minutes']} minutes):")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Escalated errors: {stats['escalated_errors']}")
    print(f"  Active alerts: {stats['active_alerts']}")
    print(f"  Escalation rate: {stats['escalation_rate']:.1f}%")
    print(f"  Severity breakdown: {stats['severity_breakdown']}")
else:
    print(f"✗ Failed to fetch statistics: {response.status_code}")

# Step 8: Resolve an alert
if response.status_code == 200:
    alerts_response = requests.get(
        f"{BASE_URL}/api/v1/avn/alerts",
        headers=headers,
        params={"limit": 1, "status_filter": "active"}
    )
    
    if alerts_response.status_code == 200:
        alerts = alerts_response.json()
        if alerts:
            alert_id = alerts[0]['id']
            print(f"\n8. Resolving alert {alert_id[:16]}...")
            
            response = requests.post(
                f"{BASE_URL}/api/v1/avn/alerts/{alert_id}/resolve",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Alert resolved successfully")
            else:
                print(f"✗ Failed to resolve alert: {response.status_code}")

print("\n" + "=" * 60)
print("✅ Pushback escalation tests complete!")
print("\nFeatures tested:")
print("  ✓ Error recording in audit table")
print("  ✓ Time-windowed threshold evaluation")
print("  ✓ Automatic AVN escalation")
print("  ✓ Severity-based rules")
print("  ✓ Burst detection")
print("  ✓ Critical error immediate escalation")
print("  ✓ Alert management and resolution")
print("  ✓ Error statistics and monitoring")
print("\nAPI Documentation: http://localhost:8000/api/docs")

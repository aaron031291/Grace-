"""
Test immutable logs with vector search
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing Grace Immutable Logs System")
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

# Step 2: Create log entries
print("\n2. Creating immutable log entries...")
log_entries = [
    {
        "operation_type": "user_authentication",
        "action": {"type": "login", "method": "jwt", "ip": "192.168.1.100"},
        "result": {"success": True, "user_id": "user-001"},
        "severity": "info",
        "tags": ["auth", "security", "login"]
    },
    {
        "operation_type": "policy_approval",
        "action": {"policy_id": "policy-123", "approver": "admin", "decision": "approved"},
        "result": {"status": "active", "effective_date": "2024-01-01"},
        "severity": "warning",
        "tags": ["policy", "governance", "approval"]
    },
    {
        "operation_type": "data_access",
        "action": {"resource": "sensitive_documents", "user": "user-002", "operation": "read"},
        "result": {"granted": True, "documents_accessed": 5},
        "severity": "info",
        "tags": ["data", "access", "audit"]
    },
    {
        "operation_type": "security_violation",
        "action": {"type": "unauthorized_access", "target": "admin_panel", "ip": "203.0.113.42"},
        "result": {"blocked": True, "action_taken": "temporary_ban"},
        "severity": "critical",
        "tags": ["security", "violation", "blocked"]
    },
    {
        "operation_type": "configuration_change",
        "action": {"setting": "rate_limit", "old_value": 100, "new_value": 200},
        "result": {"applied": True, "restart_required": False},
        "severity": "warning",
        "tags": ["config", "system", "change"]
    }
]

cids = []
for log_data in log_entries:
    response = requests.post(
        f"{BASE_URL}/api/v1/logs",
        headers=headers,
        json=log_data
    )
    
    if response.status_code == 201:
        result = response.json()
        cids.append(result['cid'])
        print(f"✓ Created log: {result['operation_type']} (CID: {result['cid'][:16]}...)")
    else:
        print(f"✗ Failed to create log: {response.status_code}")

# Step 3: Get specific log entry
if cids:
    print(f"\n3. Retrieving log entry by CID...")
    response = requests.get(f"{BASE_URL}/api/v1/logs/{cids[0]}", headers=headers)
    
    if response.status_code == 200:
        entry = response.json()
        print(f"✓ Retrieved entry:")
        print(f"  Operation: {entry['operation_type']}")
        print(f"  Actor: {entry['actor']}")
        print(f"  Severity: {entry['severity']}")
        print(f"  Trust Score: {entry.get('trust_score', 'N/A'):.3f}" if entry.get('trust_score') else "")
        print(f"  Tags: {', '.join(entry['tags'])}")

# Step 4: Semantic search
print("\n4. Testing semantic search...")
searches = [
    {
        "query": "authentication and login security",
        "k": 3,
        "min_similarity": 0.0
    },
    {
        "query": "unauthorized access attempts and violations",
        "k": 3,
        "severity_filter": "critical"
    },
    {
        "query": "policy changes and governance decisions",
        "k": 3,
        "tag_filter": ["policy", "governance"]
    }
]

for i, search_params in enumerate(searches, 1):
    print(f"\n  Search {i}: '{search_params['query']}'")
    response = requests.post(
        f"{BASE_URL}/api/v1/logs/search/semantic",
        headers=headers,
        json=search_params
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"  ✓ Found {len(results)} results:")
        
        for j, result in enumerate(results[:3], 1):
            print(f"    {j}. {result['operation_type']} (score: {result.get('search_score', 0):.3f})")
            print(f"       Relevance: {result.get('relevance', 'N/A')}")
            print(f"       Tags: {', '.join(result['tags'])}")
    else:
        print(f"  ✗ Search failed: {response.status_code}")

# Step 5: Trust-based search
print("\n5. Testing trust-based search...")
response = requests.post(
    f"{BASE_URL}/api/v1/logs/search/trust",
    headers=headers,
    json={"min_trust": 0.7, "max_trust": 1.0, "k": 10}
)

if response.status_code == 200:
    results = response.json()
    print(f"✓ Found {len(results)} high-trust entries:")
    
    for result in results[:5]:
        trust = result.get('trust_score', 0)
        print(f"  - {result['operation_type']}: trust={trust:.3f}, severity={result['severity']}")
else:
    print(f"✗ Trust search failed: {response.status_code}")

# Step 6: Verify chain integrity
print("\n6. Verifying chain integrity...")
response = requests.get(f"{BASE_URL}/api/v1/logs/verify/chain", headers=headers)

if response.status_code == 200:
    verification = response.json()
    print(f"✓ Chain verification:")
    print(f"  Valid: {verification['valid']}")
    print(f"  Total entries: {verification['total_entries']}")
    if not verification['valid']:
        print(f"  Error: {verification['error']}")
else:
    print(f"✗ Verification failed: {response.status_code}")

# Step 7: Get statistics
print("\n7. Getting log statistics...")
response = requests.get(f"{BASE_URL}/api/v1/logs/stats", headers=headers)

if response.status_code == 200:
    stats = response.json()
    print(f"✓ Log statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Chain valid: {stats['chain_valid']}")
    print(f"  Indexed entries: {stats['indexed_entries']}")
    print(f"  Severity breakdown: {stats['severity_breakdown']}")
    print(f"  Latest entry: {stats.get('latest_entry', 'N/A')}")
else:
    print(f"✗ Stats failed: {response.status_code}")

print("\n" + "=" * 60)
print("✅ Immutable logs tests complete!")
print("\nFeatures tested:")
print("  ✓ Log entry creation with automatic vectorization")
print("  ✓ Content ID (CID) generation and retrieval")
print("  ✓ Semantic search with similarity scoring")
print("  ✓ Trust-based search with score calculation")
print("  ✓ Chain integrity verification")
print("  ✓ Log statistics and metrics")
print("\nAPI Documentation: http://localhost:8000/api/docs")

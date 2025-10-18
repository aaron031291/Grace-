"""
Test script for document and search functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing Grace Document & Search System")
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

# Step 2: Check system info
print("\n2. Checking system info...")
response = requests.get(f"{BASE_URL}/api/v1/documents/system/info", headers=headers)
if response.status_code == 200:
    info = response.json()
    print(f"✓ Embedding provider: {info['embedding']['provider']}")
    print(f"  Dimension: {info['embedding']['dimension']}")
    print(f"  Vector store: {info['vector_store']['store_type']}")
    print(f"  Documents indexed: {info['vector_store']['count']}")

# Step 3: Create test documents
print("\n3. Creating test documents...")
test_docs = [
    {
        "title": "Machine Learning Basics",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "tags": ["AI", "ML", "education"],
        "is_public": False
    },
    {
        "title": "Python Programming Guide",
        "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "tags": ["Python", "programming", "tutorial"],
        "is_public": False
    },
    {
        "title": "Database Design Principles",
        "content": "Database design involves creating a detailed data model of a database. This includes defining tables, columns, data types, and relationships between tables. Good database design ensures data integrity, reduces redundancy, and improves query performance.",
        "tags": ["database", "design", "SQL"],
        "is_public": False
    }
]

doc_ids = []
for doc in test_docs:
    response = requests.post(
        f"{BASE_URL}/api/v1/documents",
        headers=headers,
        json=doc
    )
    if response.status_code == 201:
        created = response.json()
        doc_ids.append(created['id'])
        print(f"✓ Created: {created['title']} (indexed: {created['is_indexed']})")
    else:
        print(f"✗ Failed to create document: {response.status_code}")

# Step 4: List documents
print("\n4. Listing documents...")
response = requests.get(f"{BASE_URL}/api/v1/documents", headers=headers)
if response.status_code == 200:
    documents = response.json()
    print(f"✓ Found {len(documents)} documents")
    for doc in documents:
        print(f"  - {doc['title']} ({doc['word_count']} words)")

# Step 5: Search documents
print("\n5. Searching documents...")
searches = [
    "artificial intelligence and learning",
    "programming languages",
    "data storage and tables"
]

for query in searches:
    print(f"\n   Query: '{query}'")
    response = requests.post(
        f"{BASE_URL}/api/v1/documents/search",
        headers=headers,
        json={"query": query, "k": 3}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"   Found {results['total_results']} results:")
        for i, result in enumerate(results['results'][:3], 1):
            print(f"   {i}. {result['title']} (score: {result['score']:.3f})")
            print(f"      Preview: {result['content_preview'][:80]}...")
    else:
        print(f"   ✗ Search failed: {response.status_code}")

# Step 6: Get specific document
if doc_ids:
    print(f"\n6. Retrieving document {doc_ids[0]}...")
    response = requests.get(
        f"{BASE_URL}/api/v1/documents/{doc_ids[0]}",
        headers=headers
    )
    if response.status_code == 200:
        doc = response.json()
        print(f"✓ Retrieved: {doc['title']}")
        print(f"  Word count: {doc['word_count']}")
        print(f"  View count: {doc['view_count']}")
        print(f"  Tags: {', '.join(doc['tags'])}")

# Step 7: Update document
if doc_ids:
    print(f"\n7. Updating document {doc_ids[0]}...")
    response = requests.put(
        f"{BASE_URL}/api/v1/documents/{doc_ids[0]}",
        headers=headers,
        json={
            "content": "Updated content: Machine learning and deep learning are transforming how we build intelligent systems.",
            "tags": ["AI", "ML", "deep-learning", "updated"]
        }
    )
    if response.status_code == 200:
        print("✓ Document updated and re-indexed")

# Step 8: Delete a document
if len(doc_ids) > 1:
    print(f"\n8. Deleting document {doc_ids[-1]}...")
    response = requests.delete(
        f"{BASE_URL}/api/v1/documents/{doc_ids[-1]}",
        headers=headers
    )
    if response.status_code == 204:
        print("✓ Document deleted")

print("\n" + "=" * 60)
print("✅ Document and search tests complete!")
print("\nAPI Documentation: http://localhost:8000/api/docs")

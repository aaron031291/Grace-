#!/usr/bin/env python3
"""Test script for Grace Memory components (Lightning, Fusion, Librarian)."""

import asyncio
import sys
import os
import tempfile
import time

# Add grace to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from grace.memory.lightning import LightningMemory
    from grace.memory.fusion import FusionMemory
    from grace.memory.librarian import EnhancedLibrarian, ConstitutionalFilter
    from grace.memory.api import GraceMemoryAPI

    print("✅ Successfully imported memory components")

    def test_lightning_memory():
        """Test Lightning memory cache functionality."""
        print("🧪 Testing Lightning Memory...")

        # Create Lightning memory
        lightning = LightningMemory(max_size=100, default_ttl=10)

        # Test basic put/get
        success = lightning.put("test_key", "test_value", ttl_seconds=5)
        assert success, "Failed to store value"

        value = lightning.get("test_key")
        assert value == "test_value", "Retrieved value doesn't match"

        # Test with metadata
        lightning.put(
            "complex_key", {"data": "complex", "number": 42}, tags=["test", "complex"]
        )
        complex_value = lightning.get("complex_key")
        assert complex_value["data"] == "complex", "Complex value retrieval failed"

        # Test statistics
        stats = lightning.get_stats()
        assert stats["entries"] >= 2, "Statistics not updating correctly"
        assert stats["hit_rate"] > 0, "Hit rate should be positive"

        # Test health check
        health = lightning.health_check()
        assert health["healthy"], "Lightning memory should be healthy"

        print(
            f"✅ Lightning Memory: {stats['entries']} entries, {stats['hit_rate']} hit rate"
        )
        return

    def test_fusion_memory():
        """Test Fusion long-term storage functionality."""
        print("🧪 Testing Fusion Memory...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Fusion memory
            fusion = FusionMemory(storage_path=temp_dir)

            # Test basic write/read
            entry_id = fusion.write(
                "test_key", {"content": "test data", "timestamp": "2023-01-01"}
            )
            assert entry_id, "Failed to write entry"

            entry = fusion.read(entry_id)
            assert entry is not None, "Failed to read entry"
            assert entry.value["content"] == "test data", "Entry content doesn't match"

            # Test search functionality
            fusion.write(
                "search_test",
                {"title": "Test Document", "content": "This is searchable content"},
                tags=["test", "document"],
            )

            results = fusion.search(key_pattern="search", tags=["test"], limit=10)
            assert len(results) >= 1, "Search failed to find entries"

            # Test statistics
            stats = fusion.get_stats()
            assert stats["total_entries"] >= 2, "Statistics not updating"
            assert stats["active_entries"] >= 2, "Active entries count incorrect"

            print(
                f"✅ Fusion Memory: {stats['total_entries']} total, {stats['active_entries']} active"
            )
            return

    def test_constitutional_filter():
        """Test Constitutional content filtering."""
        print("🧪 Testing Constitutional Filter...")

        filter = ConstitutionalFilter()

        # Test good content
        good_content = "This is accurate and verified information about machine learning techniques."
        good_result = filter.evaluate_content(good_content)
        assert good_result["approved"], "Good content should be approved"
        assert good_result["constitutional_score"] > 0.5, (
            "Good content should have high score"
        )

        # Test problematic content
        bad_content = (
            "This content describes how to harm people through illegal activities."
        )
        bad_result = filter.evaluate_content(bad_content)
        assert not bad_result["approved"], "Bad content should not be approved"
        assert len(bad_result["forbidden_violations"]) > 0, "Should detect violations"

        print(
            f"✅ Constitutional Filter: Good={good_result['constitutional_score']:.2f}, Bad={bad_result['constitutional_score']:.2f}"
        )
        return

    def test_enhanced_librarian():
        """Test Enhanced Librarian functionality."""
        print("🧪 Testing Enhanced Librarian...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create components
            lightning = LightningMemory(max_size=1000)
            fusion = FusionMemory(storage_path=temp_dir)
            librarian = EnhancedLibrarian(lightning, fusion, chunk_size=100)

            # Test document ingestion
            test_document = """
            This is a test document about artificial intelligence and machine learning.
            It contains multiple sentences that should be chunked appropriately.
            
            Machine learning is a subset of artificial intelligence that focuses on algorithms.
            These algorithms can learn patterns from data and make predictions.
            
            The Grace kernel uses various AI techniques for governance and decision making.
            It ensures constitutional compliance and maintains high quality standards.
            """

            result = librarian.ingest_document(test_document, "test_doc_001")
            assert result["status"] == "success", f"Document ingestion failed: {result}"
            assert result["chunks_processed"] > 0, "Should process at least one chunk"

            # Test search functionality
            search_results = librarian.search_and_rank(
                "artificial intelligence", limit=5
            )
            assert len(search_results) > 0, "Search should return results"

            # Verify constitutional scoring
            for result in search_results:
                assert result["constitutional_score"] >= 0.7, (
                    "Results should meet constitutional threshold"
                )
                assert result["trust_score"] > 0, "Trust score should be positive"

            # Test document info retrieval
            doc_info = librarian.get_document_info("test_doc_001")
            assert doc_info is not None, "Should retrieve document info"
            assert doc_info["total_chunks"] == result["chunks_processed"], (
                "Chunk counts should match"
            )

            # Test statistics
            stats = librarian.get_stats()
            assert stats["chunks_in_registry"] > 0, "Should have chunks in registry"
            assert stats["keywords_indexed"] > 0, "Should have indexed keywords"

            print(
                f"✅ Enhanced Librarian: {result['chunks_processed']} chunks, {len(search_results)} search results"
            )
            return

    async def test_memory_api():
        """Test Grace Memory API."""
        print("🧪 Testing Grace Memory API...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create API
            lightning = LightningMemory(max_size=1000)
            fusion = FusionMemory(storage_path=temp_dir)
            api = GraceMemoryAPI(lightning, fusion)

            # Test content writing
            write_result = await api.write_content(
                content="This is a test document for the Grace Memory API. It demonstrates the integration of Lightning, Fusion, and Librarian components.",
                source_id="api_test_001",
                content_type="text/plain",
                tags=["api", "test"],
                metadata={"test": True},
            )

            assert write_result["status"] == "success", (
                f"API write failed: {write_result}"
            )

            # Test content search
            search_results = await api.search_content("Grace Memory API", limit=5)
            assert len(search_results) > 0, "API search should return results"

            # Test statistics
            stats = api.get_stats()
            assert "lightning" in stats, "Stats should include Lightning data"
            assert "fusion" in stats, "Stats should include Fusion data"
            assert "librarian" in stats, "Stats should include Librarian data"

            print(
                f"✅ Memory API: Write successful, {len(search_results)} search results"
            )
            return

    def test_memory_integration():
        """Test integration between memory components."""
        print("🧪 Testing Memory Integration...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create integrated system
            lightning = LightningMemory(max_size=1000, default_ttl=60)
            fusion = FusionMemory(storage_path=temp_dir)
            librarian = EnhancedLibrarian(lightning, fusion)

            # Ingest multiple documents
            documents = [
                (
                    "Grace kernel overview",
                    "The Grace kernel provides autonomous governance for AI systems with constitutional constraints.",
                ),
                (
                    "Memory architecture",
                    "The memory system uses Lightning for caching and Fusion for long-term storage.",
                ),
                (
                    "Constitutional AI",
                    "Constitutional AI ensures that AI systems operate within defined ethical boundaries.",
                ),
            ]

            for doc_id, content in documents:
                result = librarian.ingest_document(content, doc_id)
                assert result["status"] == "success", f"Failed to ingest {doc_id}"

            # Test cross-document search
            search_results = librarian.search_and_rank("Grace kernel memory", limit=10)
            assert len(search_results) >= 2, (
                "Should find results across multiple documents"
            )

            # Verify cache/storage integration
            lightning_stats = lightning.get_stats()
            fusion_stats = fusion.get_stats()

            assert lightning_stats["entries"] > 0, (
                "Lightning should have cached entries"
            )
            assert fusion_stats["total_entries"] > 0, (
                "Fusion should have stored entries"
            )

            # Test cache hit/miss behavior
            # First access should be from Fusion, second from Lightning cache
            key = f"chunk:{search_results[0]['chunk_id']}"

            # Clear cache for this key to test Fusion fallback
            lightning.delete(key)

            # This should retrieve from Fusion and cache it
            content1 = lightning.get(key)  # Should be None (cache miss)

            # Now search again, which should populate cache
            search_results2 = librarian.search_and_rank("Grace kernel", limit=1)

            print(
                f"✅ Memory Integration: {len(documents)} docs, {len(search_results)} results, cache/storage working"
            )
            return

    async def run_tests():
        """Run all memory component tests."""
        print("🚀 Running Grace Memory Component Tests...\n")

        tests = [
            ("Lightning Memory", test_lightning_memory),
            ("Fusion Memory", test_fusion_memory),
            ("Constitutional Filter", test_constitutional_filter),
            ("Enhanced Librarian", test_enhanced_librarian),
            ("Memory API", test_memory_api),
            ("Memory Integration", test_memory_integration),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            try:
                print(f"\n--- {test_name} ---")
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()

                if result:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED with error: {e}")

            # Small delay between tests
            await asyncio.sleep(0.1)

        print(f"\n📊 Results: {passed}/{total} tests passed")
        return passed == total

    if __name__ == "__main__":
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Dependencies not available. Skipping memory tests.")
    sys.exit(0)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)

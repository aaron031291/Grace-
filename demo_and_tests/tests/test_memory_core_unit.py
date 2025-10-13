import pytest
import tempfile
from grace.core.memory_core import MemoryCore
from grace.core.contracts import Experience


@pytest.mark.asyncio
def test_store_and_recall_structured_memory():
    with tempfile.NamedTemporaryFile() as tf:
        memory_core = MemoryCore(db_path=tf.name)
        content = {"foo": "bar"}
        # store_structured_memory is async; run it via asyncio
        memory_id = __import__("asyncio").run(
            memory_core.store_structured_memory("test", content)
        )
        recalled = __import__("asyncio").run(
            memory_core.recall_structured_memory(memory_id)
        )
        assert recalled is not None
        assert recalled["content"] == content


@pytest.mark.asyncio
def test_store_experience():
    from datetime import datetime
    with tempfile.NamedTemporaryFile() as tf:
        memory_core = MemoryCore(db_path=tf.name)

        exp = Experience(
            type="unit",
            component_id="test",
            context={"meta": "test"},
            outcome={"result": "ok"},
            success_score=0.9,
            timestamp=datetime.utcnow(),
        )
        memory_id = __import__("asyncio").run(memory_core.store_experience(exp))
        recalled = __import__("asyncio").run(
            memory_core.recall_structured_memory(memory_id)
        )
        assert recalled is not None
        assert recalled["content"]["component_id"] == "test"
        assert recalled["content"]["success_score"] == 0.9

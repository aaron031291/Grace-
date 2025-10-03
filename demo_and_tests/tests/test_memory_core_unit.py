import pytest
from grace.core.memory_core import MemoryCore
from grace.core.contracts import Experience

@pytest.mark.asyncio
def test_store_and_recall_structured_memory():
    memory_core = MemoryCore(db_path=':memory:')
    content = {"foo": "bar"}
    memory_id = pytest.run(memory_core.store_structured_memory("test", content))
    recalled = pytest.run(memory_core.recall_structured_memory(memory_id))
    assert recalled is not None
    assert recalled["content"] == content

@pytest.mark.asyncio
def test_store_experience():
    memory_core = MemoryCore(db_path=':memory:')
    exp = Experience(component_id="test", type="unit", success_score=0.9)
    memory_id = pytest.run(memory_core.store_experience(exp))
    recalled = pytest.run(memory_core.recall_structured_memory(memory_id))
    assert recalled is not None
    assert recalled["content"]["component_id"] == "test"
    assert recalled["content"]["success_score"] == 0.9

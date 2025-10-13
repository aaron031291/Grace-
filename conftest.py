import asyncio
import pytest

@pytest.fixture(autouse=True)
async def cleanup_asyncio_tasks():
    """Autouse fixture to cancel any pending asyncio tasks after each test.

    This helps prevent tasks created by modules under test from leaking into
    other tests and triggering "Task was destroyed but it is pending!" warnings.
    """
    yield

    # Give scheduled tasks a short moment to finish naturally
    await asyncio.sleep(0)

    current = asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]

    if not tasks:
        return

    for t in tasks:
        try:
            t.cancel()
        except Exception:
            pass

    for t in tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            # swallow any errors raised by tasks during cancellation
            pass

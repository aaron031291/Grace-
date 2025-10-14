import os
import pytest


def pytest_collection_modifyitems(config, items):
    run_e2e = os.environ.get("RUN_E2E") == "1"
    if not run_e2e:
        skip_e2e = pytest.mark.skip(reason="E2E tests disabled (set RUN_E2E=1 to enable)")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)

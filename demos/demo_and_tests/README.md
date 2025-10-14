# Grace Demo and Tests

This directory contains all demonstration and testing code for the Grace system, organized for better maintainability.

## Structure

```
demo_and_tests/
├── demos/          # Demonstration scripts
├── tests/          # All test files and reports
├── __init__.py     # Package initializer
└── README.md       # This file
```

## Demos Directory (`demos/`)

Contains demonstration scripts that showcase various features and capabilities of the Grace governance system:

- `demo_clarity_framework.py` - Clarity framework demonstrations
- `demo_gme_workflows.py` - Grace Message Envelope workflow examples
- `demo_ingress_kernel.py` - Ingress kernel functionality demo
- `demo_integrated_services.py` - Integrated services demonstrations
- `demo_mldl_kernel.py` - ML/DL kernel examples
- `demo_multi_os_kernel.py` - Multi-OS kernel capabilities
- `demo_orb_interface.py` - ORB interface demonstrations
- `demo_resilience_kernel.py` - Resilience kernel examples

## Tests Directory (`tests/`)

Contains all testing code including:

### Unit Tests
- `test_governance_kernel.py` - Main governance kernel tests
- `test_event_mesh.py` - Event mesh functionality tests
- `test_memory_components.py` - Memory system tests
- `test_orchestration_kernel.py` - Orchestration tests
- `test_resilience_kernel.py` - Resilience system tests
- And more...

### Integration Tests
- `integration_tests.py` - Comprehensive integration test suite
- `test_grace_integration.py` - Grace system integration tests

### End-to-End Tests
- `e2e/` - End-to-end test scenarios
- `smoke_tests.py` - Quick smoke tests

### Test Reports
- Various `.json` files containing test execution reports and system health reports

## Running Tests

From the project root directory:

```bash
# Run a specific test
python demo_and_tests/tests/test_governance_kernel.py

# Run integration tests
python demo_and_tests/tests/integration_tests.py

# Run smoke tests
python demo_and_tests/tests/smoke_tests.py
```

## Running Demos

From the project root directory:

```bash
# Run a specific demo
python demo_and_tests/demos/demo_event_mesh.py

# Run governance demo
python demo_and_tests/demos/demo_governance_kernel.py
```

## Notes

- All files have been updated with correct import paths after the reorganization
- The project structure now consolidates all demo and test code in one location
- Original functionality is preserved while providing better organization
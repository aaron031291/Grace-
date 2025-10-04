# CI Coverage Enforcement Script
pytest --cov=grace --cov=backend --cov=scripts --cov=demo_and_tests --cov-report=term-missing --cov-fail-under=80

# Add this step to your CI pipeline to enforce ≥80% coverage

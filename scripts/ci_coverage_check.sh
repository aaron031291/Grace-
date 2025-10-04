# CI Coverage Enforcement Script

COVERAGE_THRESHOLD=80

# Run pytest with coverage
pytest --cov=grace --cov=backend --cov=scripts --cov=demo_and_tests --cov-report=term-missing --cov-report=xml

# Extract coverage percentage
COVERAGE=$(grep 'line-rate' coverage.xml | head -1 | sed -E 's/.*line-rate="([0-9.]+)".*/\1/')
COVERAGE_PERCENT=$(echo "$COVERAGE * 100" | bc | awk '{print int($1+0.5)}')

if [ "$COVERAGE_PERCENT" -lt "$COVERAGE_THRESHOLD" ]; then
	echo "Test coverage $COVERAGE_PERCENT% is below threshold $COVERAGE_THRESHOLD%. Failing CI."
	exit 1
else
	echo "Test coverage $COVERAGE_PERCENT% meets threshold $COVERAGE_THRESHOLD%."
fi

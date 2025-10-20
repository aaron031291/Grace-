# CI Security & Compliance Checks


# Bandit - Python SAST
echo "Running Bandit SAST scan..."
bandit -r grace backend scripts > bandit_report.txt || true

# pip-audit - Dependency scan
echo "Running pip-audit dependency scan..."
pip-audit -r requirements.txt > pip_audit_report.txt || true

# Trivy - Container image scan (example for Dockerfile)
echo "Running Trivy image scan..."
trivy image grace_api:latest > trivy_report.txt || true

echo "Security checks complete. See bandit_report.txt, pip_audit_report.txt, trivy_report.txt."

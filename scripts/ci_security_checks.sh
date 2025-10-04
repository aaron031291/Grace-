# CI Security & Compliance Checks

# Bandit - Python SAST
bandit -r grace backend scripts

# pip-audit - Dependency scan
pip-audit -r requirements.txt

# Trivy - Container image scan (example for Dockerfile)
trivy image grace_api:latest

# To use in CI, add these steps to your pipeline config (GitHub Actions, GitLab CI, etc.)

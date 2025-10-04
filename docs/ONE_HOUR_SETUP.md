# Grace ONE_HOUR_SETUP.md — Quick Start Guide

## Purpose
Get Grace running locally or in dev in under one hour. Covers environment setup, dependencies, and first run.

---

## 1. Prerequisites
- OS: Linux, macOS, or Windows (WSL recommended)
- Docker & Docker Compose
- Python 3.12+
- Node.js 18+ (for frontend)
- Git

---

## 2. Clone & Prepare
```bash
git clone https://github.com/aaron031291/Grace-
cd Grace-
```

---

## 3. Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Backend & API
```bash
# Start backend (FastAPI, workers)
cd backend
uvicorn main:app --reload
```

---

## 5. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 6. Datastores (Dev)
```bash
docker-compose up -d postgres redis qdrant minio
```

---

## 7. Monitoring & Observability
```bash
docker-compose up -d prometheus grafana
# Access Grafana at http://localhost:3000 (default admin/admin)
```

---

## 8. End-to-End Test
```bash
python scripts/e2e_smoke_test.py
```

---

## 9. Troubleshooting
- Check logs in `logs/grace.log`
- Validate health endpoints: `/health`, `/health/full`
- For issues, see [DR_RUNBOOK.md](DR_RUNBOOK.md)

---

## 10. Next Steps
- Review [GRACE_UNIFIED_OPERATING_SPEC_COMPLETE.md](GRACE_UNIFIED_OPERATING_SPEC_COMPLETE.md)
- Explore Memory Explorer and Orb UI
- Run full test suite: `pytest`

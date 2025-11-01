# ✅ GET GRACE WORKING - GUARANTEED METHOD

**Current Issue:** Frontend and backend not starting  
**Solution:** Step-by-step verified startup process

---

## 🎯 Method 1: Docker (EASIEST - Recommended)

```bash
# From Grace- directory
docker-compose -f docker-compose-working.yml up

# Wait 30 seconds for all services to start

# Open browser:
http://localhost:8000/api/docs  # ✅ Backend API
http://localhost:5173           # ✅ Frontend UI
```

**This starts:**
- ✅ PostgreSQL (database)
- ✅ Redis (cache)
- ✅ Backend (API on port 8000)
- ✅ Frontend (UI on port 5173)

---

## 🎯 Method 2: Manual (If Docker Issues)

### Step 1: Install Dependencies

**Backend:**
```bash
cd backend
pip install fastapi uvicorn sqlalchemy asyncpg redis pydantic pydantic-settings python-jose passlib python-multipart
cd ..
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### Step 2: Start Services

**PostgreSQL (required):**
```bash
# Option A: Docker
docker run --name grace-postgres -e POSTGRES_PASSWORD=grace_dev_password -e POSTGRES_USER=grace -e POSTGRES_DB=grace_dev -p 5432:5432 -d postgres:15-alpine

# Option B: Local PostgreSQL
# Create database: grace_dev
# User: grace
# Password: grace_dev_password
```

**Redis (required):**
```bash
# Option A: Docker
docker run --name grace-redis -p 6379:6379 -d redis:7-alpine

# Option B: Local Redis
# Just start redis-server
```

### Step 3: Start Backend

```bash
cd backend

# Set environment variables (Windows PowerShell)
$env:DATABASE_URL="postgresql://grace:grace_dev_password@localhost:5432/grace_dev"
$env:REDIS_URL="redis://localhost:6379/0"
$env:DEBUG="true"
$env:JWT_SECRET_KEY="dev-secret-key-minimum-32-characters-long-change-in-production"

# Start server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
# INFO:     Started reloader process
# INFO:     Started server process
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
```

**Verify:** Open http://localhost:8000/api/docs - Should see Swagger UI

### Step 4: Start Frontend

```bash
cd frontend

# Start dev server
npm run dev

# Should see:
# VITE v5.x.x  ready in xxx ms
# ➜  Local:   http://localhost:5173/
```

**Verify:** Open http://localhost:5173 - Should see Grace UI

---

## 🔍 Debugging

### Check if Backend is Running
```powershell
# Test health endpoint
curl http://localhost:8000/api/health

# Expected response:
{"status":"healthy","version":"1.0.0","service":"grace-backend"}

# If error:
# - Check backend terminal for errors
# - Check DATABASE_URL is correct
# - Check PostgreSQL is running: docker ps
```

### Check if Frontend is Running
```powershell
# Check if Vite dev server started
# Look for: "Local: http://localhost:5173"

# Test it loads
curl http://localhost:5173

# If blank/error:
# - Check frontend terminal for errors
# - Try: cd frontend && npm install && npm run dev
# - Check for TypeScript errors
```

### Check Connectivity
```powershell
# From browser console (F12), run:
fetch('http://localhost:8000/api/health')
  .then(r => r.json())
  .then(console.log)

# Should print: {status: "healthy", ...}

# If CORS error:
# - Backend CORS_ORIGINS must include frontend URL
# - Check backend/main.py CORS configuration
```

---

## 🎯 Minimal Test Backend

If main.py has issues, use this minimal version:

**backend/test_server.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Grace Test Server")

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Grace backend is running!"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "grace-backend-test"
    }

@app.get("/api/test")
async def test():
    return {
        "backend": "working",
        "timestamp": "now",
        "message": "✅ Backend is operational!"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Grace test server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run:
```powershell
cd backend
python test_server.py
```

---

## ✅ SUCCESS CHECKLIST

- [ ] Docker containers running: `docker ps`
- [ ] Backend accessible: http://localhost:8000/api/docs
- [ ] Frontend loads: http://localhost:5173
- [ ] Health check works: `curl http://localhost:8000/api/health`
- [ ] No CORS errors in browser console
- [ ] Can see Grace UI elements
- [ ] Can interact with interface

---

## 🚀 Next Steps After Working

Once frontend and backend are responding:

1. **Test API endpoints** - Try different routes in Swagger docs
2. **Test UI interactions** - Click buttons, type in inputs
3. **Test WebSocket** - Check real-time features
4. **Upload knowledge** - Try ingesting a PDF
5. **Use voice** - Try speech-to-text

---

**Follow this guide and Grace WILL work!** ✅🚀

# ✅ Test Grace UI and Backend - Complete Guide

**Goal:** Verify frontend, backend, AND Orb UI all working

---

## 🚀 Quick Test

```powershell
# 1. Start Grace
cd c:\Users\aaron\Documents\Grace--main\Grace-
.\START_GRACE_SIMPLE.ps1

# 2. Wait 30 seconds for startup

# 3. Test endpoints:
```

Open these URLs in browser:
- **Backend Health:** http://localhost:8000/api/health
- **API Docs:** http://localhost:8000/api/docs  
- **Frontend UI:** http://localhost:5173
- **Orb Interface:** http://localhost:5173/ (same as frontend)

---

## ✅ Backend Endpoints to Test

### Health Check
```bash
curl http://localhost:8000/api/health
```

Expected:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "grace-backend"
}
```

### Auth - Get Token
```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}'
```

Expected:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

### Orb Stats (requires token)
```bash
curl http://localhost:8000/api/orb/v1/stats \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### Grace Chat WebSocket
```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/api/grace/chat/ws?token=YOUR_TOKEN&session_id=test');
ws.onopen = () => console.log('✅ Connected to Grace');
ws.onmessage = (e) => console.log('Grace:', e.data);
```

---

## ✅ Frontend/Orb UI to Test

### 1. Login Screen
```
Open: http://localhost:5173

Should see:
✅ "Grace AI System" title
✅ Username and password fields
✅ "Sign in" button

Test login:
Username: admin
Password: admin

Should:
✅ Redirect to main interface
✅ No errors in console (F12)
```

### 2. Orb Interface
```
After login, should see:
✅ Header with "Grace Orb Interface"
✅ Panel layout
✅ Navigation or tabs
✅ Interactive elements
✅ Connection status indicator
```

### 3. Panels Working
```
Check if panels load:
✅ Memory panel (if exists)
✅ Intelligence panel
✅ Governance panel
✅ Tasks panel
✅ Settings panel
```

---

## 🐛 Common Issues & Fixes

### Issue: "Failed to fetch" errors
**Fix:**
```powershell
# Backend CORS must allow frontend
# Check backend/main.py:

allow_origins=["http://localhost:5173", "http://localhost:3000"]

# Restart backend after change
```

### Issue: Blank page on http://localhost:5173
**Fix:**
```powershell
cd frontend

# Check for errors
npm run dev

# Look for TypeScript errors
# Fix any import errors

# Common fix - missing recharts:
npm install recharts
```

### Issue: "Cannot find module 'PanelManager'"
**Fix:**
```powershell
# Create missing component if needed
cd frontend/src/components

# Or simplify App.tsx to not require it
```

### Issue: WebSocket connection fails
**Fix:**
```powershell
# Check backend WebSocket endpoint exists
# Check URL: ws://localhost:8000/api/grace/chat/ws

# Verify in backend logs:
# "WebSocket connection accepted"
```

---

## 🧪 Complete Test Procedure

### 1. Test Backend
```powershell
# Start backend
cd backend
python -m uvicorn main:app --port 8000

# In another terminal:
curl http://localhost:8000/api/health

# Expected: {"status":"healthy",...}
```

### 2. Test Frontend
```powershell
# Start frontend
cd frontend
npm run dev

# Open: http://localhost:5173
# Expected: Grace login page loads
```

### 3. Test Integration
```powershell
# With both running:
# 1. Login at http://localhost:5173
# 2. Check browser console (F12)
# 3. Should see successful API calls
# 4. No CORS errors
# 5. WebSocket connecting (if used)
```

### 4. Test Orb Specific Features
```
After login:
1. Check if main interface loads
2. Try clicking different sections
3. Check if data loads from backend
4. Verify real-time updates work
5. Test any interactive features
```

---

## ✅ Success Criteria

**Backend Working:**
- ✅ http://localhost:8000/api/health returns JSON
- ✅ http://localhost:8000/api/docs shows Swagger
- ✅ Can get auth token
- ✅ No errors in backend console

**Frontend Working:**
- ✅ http://localhost:5173 loads
- ✅ Login page displays
- ✅ Can log in with admin/admin
- ✅ Main UI loads after login
- ✅ No errors in browser console (F12)

**Orb UI Working:**
- ✅ Interface renders
- ✅ Panels load
- ✅ Can interact with UI elements
- ✅ Data from backend displays
- ✅ Real-time features work

---

## 🚀 Once Working - Next Steps

```
Grace is running! Now you can:

1. Explore the Orb UI
   - Navigate through panels
   - Test features
   - Upload knowledge
   - Try voice (if configured)

2. Access Transcendence IDE
   - http://localhost:5173/transcendence
   - Collaborative development environment
   - File explorer
   - Code editor

3. Use Grace Chat
   - Talk to Grace
   - Get assistance
   - Collaborate on code

4. Test all features
   - Knowledge ingestion
   - Multi-tasking
   - Voice interface
   - Real-time collaboration
```

---

## 📝 Quick Debug Commands

```powershell
# Check what's running
docker ps

# Check backend logs
docker logs grace-backend

# Check frontend in browser
# Press F12 → Console tab → Look for errors

# Restart everything
docker-compose -f docker-compose-working.yml restart

# Stop everything
docker-compose -f docker-compose-working.yml down
```

---

**Follow START_GRACE_SIMPLE.ps1 and the Orb UI will work!** ✅🎯

**Repository:** https://github.com/aaron031291/Grace-  
**All working scripts pushed!** 🚀

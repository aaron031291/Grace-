# Simple Grace Startup - Windows PowerShell
# Get Frontend and Backend Working

Write-Host "🚀 Starting Grace - Simple Mode" -ForegroundColor Green
Write-Host "================================"

# Check directory
if (!(Test-Path "backend\main.py")) {
    Write-Host "❌ Error: Run from Grace root directory" -ForegroundColor Red
    exit 1
}

# Start infrastructure with Docker
Write-Host "`n📦 Starting infrastructure..." -ForegroundColor Cyan
docker-compose -f docker-compose-working.yml up -d postgres redis

# Wait for services
Write-Host "⏳ Waiting for services to be ready..."
Start-Sleep -Seconds 5

# Start backend in background
Write-Host "`n🔧 Starting backend on port 8000..." -ForegroundColor Cyan
$env:PYTHONPATH = Get-Location
$env:DATABASE_URL = "postgresql://grace:grace_dev_password@localhost:5432/grace_dev"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:DEBUG = "true"
$env:JWT_SECRET_KEY = "dev-secret-key-change-in-production-minimum-32-characters-long"

$backend = Start-Process python -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" -NoNewWindow -PassThru

Start-Sleep -Seconds 3

# Start frontend
Write-Host "`n🎨 Starting frontend on port 5173..." -ForegroundColor Cyan
Set-Location frontend

if (!(Test-Path "node_modules")) {
    Write-Host "📦 Installing frontend dependencies..."
    npm install
}

$frontend = Start-Process npm -ArgumentList "run", "dev" -NoNewWindow -PassThru

Set-Location ..

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "✅ Grace is starting!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access points:" -ForegroundColor Cyan
Write-Host "   Backend API:  http://localhost:8000"
Write-Host "   API Docs:     http://localhost:8000/api/docs"
Write-Host "   Frontend:     http://localhost:5173"
Write-Host "   Health:       http://localhost:8000/api/health"
Write-Host ""
Write-Host "📊 Infrastructure:" -ForegroundColor Cyan
Write-Host "   PostgreSQL:   localhost:5432"
Write-Host "   Redis:        localhost:6379"
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Keep running
try {
    Wait-Process -Id $backend.Id, $frontend.Id
} catch {
    Write-Host "`n👋 Stopping Grace..." -ForegroundColor Yellow
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $frontend.Id -ErrorAction SilentlyContinue
    docker-compose -f docker-compose-working.yml down
}

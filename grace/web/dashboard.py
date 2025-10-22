"""
Grace Web Dashboard - Interactive UI
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Get template directory
template_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
template_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(template_dir))


def create_dashboard_app() -> FastAPI:
    """Create dashboard application"""
    app = FastAPI(title="Grace Dashboard")
    
    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home(request: Request):
        """Main dashboard page"""
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "title": "Grace Dashboard"
        })
    
    @app.get("/kernels", response_class=HTMLResponse)
    async def kernels_page(request: Request):
        """Kernels management page"""
        return templates.TemplateResponse("kernels.html", {
            "request": request,
            "title": "Kernels - Grace Dashboard"
        })
    
    @app.get("/metrics", response_class=HTMLResponse)
    async def metrics_page(request: Request):
        """Metrics and KPIs page"""
        return templates.TemplateResponse("metrics.html", {
            "request": request,
            "title": "Metrics - Grace Dashboard"
        })
    
    return app

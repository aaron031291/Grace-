#!/usr/bin/env python3
"""
Grace Complete Startup Script

ONE command to start everything:
- Backend API (FastAPI)
- Frontend Dev Server (React + Vite)
- Orb UI
- Breakthrough System
- MCP Server
- Expert Code Generator
- Collaborative Features

Grace becomes fully operational as an expert coding AI.
"""

import asyncio
import subprocess
import sys
import logging
from pathlib import Path
import signal
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraceCompleteSystem:
    """Orchestrates all Grace services"""
    
    def __init__(self):
        self.processes = []
        self.services = {
            "backend": None,
            "frontend": None,
            "breakthrough": None,
            "mcp": None
        }
        
    async def start_all(self):
        """Start all Grace services"""
        print("\n" + "🌟 "*30)
        print("GRACE AI - COMPLETE SYSTEM STARTUP")
        print("🌟 "*30 + "\n")
        
        print("Starting all services...")
        print("  • Backend API (FastAPI)")
        print("  • Frontend UI (React + Vite)")
        print("  • Breakthrough System")
        print("  • MCP Server")
        print("  • Expert Code Generator")
        print("")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # 1. Start Backend
            await self._start_backend()
            await asyncio.sleep(2)
            
            # 2. Start Frontend
            await self._start_frontend()
            await asyncio.sleep(2)
            
            # 3. Initialize Expert Systems
            await self._initialize_expert_systems()
            
            # 4. Start Breakthrough (optional continuous mode)
            # await self._start_breakthrough()
            
            # Print status
            self._print_status()
            
            # Keep running
            print("\n✅ All services started!")
            print("\n📊 Access Points:")
            print("   Backend API:  http://localhost:8000")
            print("   API Docs:     http://localhost:8000/api/docs")
            print("   Frontend:     http://localhost:5173")
            print("   Orb UI:       http://localhost:5173/orb")
            print("")
            print("Press Ctrl+C to stop all services\n")
            
            # Wait
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Shutting down...")
            await self.stop_all()
    
    async def _start_backend(self):
        """Start FastAPI backend"""
        print("🚀 Starting Backend API...")
        
        try:
            backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "backend.main:app", 
                 "--reload", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            self.services["backend"] = backend_process
            self.processes.append(backend_process)
            
            print("   ✅ Backend started on http://localhost:8000")
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            print(f"   ❌ Backend failed: {e}")
    
    async def _start_frontend(self):
        """Start React frontend"""
        print("🎨 Starting Frontend UI...")
        
        try:
            frontend_dir = Path(__file__).parent / "frontend"
            
            # Check if node_modules exists
            if not (frontend_dir / "node_modules").exists():
                print("   📦 Installing frontend dependencies...")
                subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    check=True
                )
            
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=frontend_dir
            )
            
            self.services["frontend"] = frontend_process
            self.processes.append(frontend_process)
            
            print("   ✅ Frontend started on http://localhost:5173")
            
        except Exception as e:
            logger.error(f"Failed to start frontend: {e}")
            print(f"   ❌ Frontend failed: {e}")
    
    async def _initialize_expert_systems(self):
        """Initialize Grace's expert knowledge systems"""
        print("🧠 Initializing Expert Systems...")
        
        try:
            # Initialize expert knowledge
            from grace.knowledge.expert_system import get_expert_system
            expert_sys = get_expert_system()
            
            summary = expert_sys.get_all_expertise_summary()
            print(f"   ✅ Expert System loaded ({summary['total_domains']} domains)")
            print(f"      Average proficiency: {summary['avg_proficiency']:.0%}")
            
            # Initialize expert code generator
            from grace.intelligence.expert_code_generator import get_expert_code_generator
            code_gen = get_expert_code_generator()
            
            print(f"   ✅ Expert Code Generator ready")
            
            # Initialize MCP server
            from grace.mcp.mcp_server import get_mcp_server
            mcp = get_mcp_server()
            tools = mcp.list_tools()
            
            print(f"   ✅ MCP Server ready ({len(tools)} tools)")
            
            # Initialize collaborative generator
            from grace.mtl.collaborative_code_gen import CollaborativeCodeGenerator
            collab = CollaborativeCodeGenerator()
            
            print(f"   ✅ Collaborative Code Gen ready")
            
            print("\n   Grace is now an expert in:")
            for domain in summary['domains']:
                print(f"      • {domain['domain']}: {domain['proficiency']:.0%} proficiency")
            
        except Exception as e:
            logger.error(f"Failed to initialize expert systems: {e}")
            print(f"   ⚠️  Expert systems: {e}")
    
    async def _start_breakthrough(self):
        """Start breakthrough system in continuous mode"""
        print("🔄 Starting Breakthrough System...")
        
        try:
            from grace.core.breakthrough import BreakthroughSystem
            
            breakthrough = BreakthroughSystem()
            await breakthrough.initialize()
            
            # Start in background
            asyncio.create_task(
                breakthrough.run_continuous_improvement(interval_hours=24)
            )
            
            print("   ✅ Breakthrough System running (continuous mode)")
            
        except Exception as e:
            logger.error(f"Failed to start breakthrough: {e}")
            print(f"   ⚠️  Breakthrough: {e}")
    
    def _print_status(self):
        """Print service status"""
        print("\n" + "="*70)
        print("SERVICE STATUS")
        print("="*70 + "\n")
        
        for service, process in self.services.items():
            if process and process.poll() is None:
                print(f"   ✅ {service.capitalize()}: Running")
            else:
                print(f"   ❌ {service.capitalize()}: Not running")
    
    async def stop_all(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")
        
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("   ✅ All services stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\nReceived shutdown signal...")
        asyncio.create_task(self.stop_all())
        sys.exit(0)


async def main():
    """Main entry point"""
    system = GraceCompleteSystem()
    await system.start_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")

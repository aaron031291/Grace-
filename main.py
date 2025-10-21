"""
Main entry point for Grace
"""

import asyncio

def run_service():
    """Run Grace in service mode"""
    from grace.core.unified_service import create_unified_app
    import uvicorn
    
    app = create_unified_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

async def run_demo(demo_name: str):
    """Run a specific demo"""
    if demo_name == "multi_os":
        from grace.demo.multi_os_kernel import demo_multi_os_kernel
        await demo_multi_os_kernel()
    elif demo_name == "mldl":
        from grace.demo.mldl_kernel import demo_mldl_kernel
        await demo_mldl_kernel()
    elif demo_name == "resilience":
        from grace.demo.resilience_kernel import demo_resilience_kernel
        await demo_resilience_kernel()
    else:
        print(f"Unknown demo: {demo_name}")
        print("Available demos: multi_os, mldl, resilience")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        asyncio.run(run_demo(demo_name))
    else:
        run_service()
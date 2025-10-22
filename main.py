"""
Main entry point for Grace
"""

import argparse
import asyncio
import sys

def run_service():
    """Run Grace in service mode"""
    # Lazy import to avoid import-time cycles
    from grace.core.unified_service import create_unified_app
    import uvicorn

    app = create_unified_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


async def run_demo(demo_name: str):
    """Run a specific demo (coroutine)"""
    # Map demo names to kernel start functions
    demos = {
        "multi_os": "grace.kernels.multi_os:start",
        "mldl": "grace.kernels.mldl:start",
        "resilience": "grace.kernels.resilience:start",
    }

    if demo_name not in demos:
        print(f"Unknown demo: {demo_name}")
        print("Available demos:", ", ".join(demos.keys()))
        return

    module_path, func = demos[demo_name].rsplit(":", 1)
    module = __import__(module_path, fromlist=[func])
    start_coro = getattr(module, func)
    await start_coro()

def main(argv=None):
    parser = argparse.ArgumentParser(prog="grace")
    parser.add_argument("mode", nargs="?", default="service", choices=["service", "demo"])
    parser.add_argument("--demo", help="Demo to run when mode=demo", default="multi_os")
    args = parser.parse_args(argv)

    if args.mode == "service":
        run_service()
    elif args.mode == "demo":
        asyncio.run(run_demo(args.demo))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Multi-OS Kernel Demo - Quick demonstration of Multi-OS capabilities.
"""
import asyncio
import sys
import os

# Add the multi_os directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_os'))

from multi_os_service import MultiOSService


async def demo_multi_os_kernel():
    """Demonstrate Multi-OS Kernel capabilities."""
    print("🚀 Multi-OS Kernel Demo")
    print("=" * 50)
    
    # Initialize the service
    print("Initializing Multi-OS Service...")
    service = MultiOSService()
    print("✓ Service ready with Linux, Windows, and macOS support\n")
    
    # Register a host
    print("📟 Registering a sample host...")
    sample_host = {
        "host_id": "demo-linux-001",
        "os": "linux",
        "arch": "x86_64", 
        "agent_version": "2.4.1",
        "capabilities": ["process", "fs", "net", "pkg", "sandbox"],
        "labels": ["region:demo", "env:development"],
        "status": "online"
    }
    
    host_id = service.registry.register_host(sample_host)
    print(f"✓ Host registered: {host_id}")
    
    # Create and place a task
    print("\n⚙️ Creating and placing a task...")
    demo_task = {
        "task_id": "demo-task-001",
        "command": "python",
        "args": ["-c", "print('Hello from Multi-OS Kernel!')"],
        "runtime": {
            "runtime": "python",
            "version": "3.11"
        },
        "constraints": {
            "os": ["linux", "macos"],
            "sandbox": "nsjail"
        }
    }
    
    hosts = service.registry.list_hosts({"status": "online"})
    placement = await service.scheduler.place(demo_task, hosts)
    
    if placement["success"]:
        print(f"✓ Task placed on host: {placement['host_id']}")
        print(f"  Score: {placement['score']:.3f}")
    else:
        print("❌ Task placement failed")
    
    # Show telemetry
    print("\n📊 Recording telemetry...")
    service.telemetry.record_metric("demo_metric", 42.0, {"component": "demo"})
    service.telemetry.record_log("INFO", "Demo execution started")
    
    kpis = service.telemetry.get_kpis()
    print(f"✓ KPIs being tracked: {len(kpis['kpis'])}")
    
    # Create snapshot
    print("\n📸 Creating system snapshot...")
    snapshot = await service.snapshot_manager.create_snapshot("agent")
    print(f"✓ Snapshot created: {snapshot['snapshot_id']}")
    
    # Show service stats
    print("\n📈 Service Statistics:")
    stats = service.get_service_stats()
    print(f"  Hosts: {stats['hosts']['total_hosts']}")
    print(f"  OS Types: {list(stats['hosts']['by_os'].keys())}")
    print(f"  Adapters: {stats['adapters']}")
    print(f"  Snapshots: {stats['snapshots']['total_snapshots']}")
    
    print("\n🎉 Demo completed successfully!")
    print("Multi-OS Kernel provides unified cross-platform execution")
    print("with intelligent placement, telemetry, and snapshot management.")


def main():
    """Run the demo."""
    try:
        asyncio.run(demo_multi_os_kernel())
        return 0
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
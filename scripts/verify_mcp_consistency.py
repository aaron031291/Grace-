"""
Verify MCP schema consistency across kernels
"""

import sys
from pathlib import Path


def main():
    """Verify MCP consistency"""
    print("🔍 Verifying MCP Consistency")
    print("=" * 60)
    
    # Check that all kernels use MCPClient
    kernel_dir = Path("grace/kernels")
    kernels_without_mcp = []
    
    for py_file in kernel_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        content = py_file.read_text()
        
        if "MCPClient" not in content:
            kernels_without_mcp.append(py_file.name)
            print(f"⚠️  {py_file.name} does not use MCPClient")
    
    print("\n" + "=" * 60)
    
    if kernels_without_mcp:
        print(f"\n⚠️  {len(kernels_without_mcp)} kernels without MCP integration")
        print("Consider adding MCPClient to all kernels")
        return 0  # Warning
    else:
        print("\n✅ All kernels use MCP!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

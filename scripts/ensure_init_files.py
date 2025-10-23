"""
Ensure all directories have __init__.py files
"""

from pathlib import Path

def ensure_init_files():
    """Create missing __init__.py files"""
    grace_dir = Path("grace")
    
    # List of directories that should have __init__.py
    directories = [
        grace_dir,
        grace_dir / "api",
        grace_dir / "api" / "v1",
        grace_dir / "auth",
        grace_dir / "config",
        grace_dir / "core",
        grace_dir / "database",
        grace_dir / "documents",
        grace_dir / "embeddings",
        grace_dir / "vectorstore",
        grace_dir / "governance",
        grace_dir / "clarity",
        grace_dir / "mldl",
        grace_dir / "avn",
        grace_dir / "orchestration",
        grace_dir / "testing",
        grace_dir / "mtl",
        grace_dir / "swarm",
        grace_dir / "transcendence",
        grace_dir / "integration",
        grace_dir / "websocket",
        grace_dir / "middleware",
        grace_dir / "observability",
        grace_dir / "memory",
        grace_dir / "trust",
        grace_dir / "events",
        grace_dir / "llm",
        grace_dir / "demo",
    ]
    
    created = []
    
    for directory in directories:
        if directory.exists():
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""\n{directory.name.capitalize()} module\n"""\n')
                created.append(str(init_file))
                print(f"Created: {init_file}")
    
    if created:
        print(f"\n✅ Created {len(created)} __init__.py files")
    else:
        print("✅ All __init__.py files already exist")
    
    return len(created)


if __name__ == "__main__":
    ensure_init_files()

"""
Setup script for Grace AI System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="grace-ai",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "demos", "demos.*"]),
    package_data={
        "grace": ["py.typed"],
    },
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "pydantic-settings>=2.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",
        "aioredis>=2.0.1",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
        "llm": [
            "llama-cpp-python>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "grace=grace.cli.commands:main",
        ],
    },
)

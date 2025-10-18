"""
Setup script for Grace AI System
"""

from setuptools import setup, find_packages

setup(
    name="grace-ai",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "grace": ["py.typed"],
    },
    include_package_data=True,
)

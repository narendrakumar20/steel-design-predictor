"""
Setup script for SteelML package.
For modern Python packaging, most configuration is in pyproject.toml.
This file provides backward compatibility.
"""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=["tests", "tests.*"]),
        include_package_data=True,
    )

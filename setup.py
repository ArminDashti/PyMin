#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyMin",
    version="1.0.0",
    author="Armin Dashti",
    author_email="pymin@example.com",
    description="A Python toolkit for various data science and AI tasks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PyMin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "pymin=PyMin.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="python, toolkit, data-science, machine-learning, ai, cli",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/PyMin/issues",
        "Source": "https://github.com/yourusername/PyMin",
        "Documentation": "https://github.com/yourusername/PyMin#readme",
    },
)

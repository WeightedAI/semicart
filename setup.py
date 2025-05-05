"""SemiCART setup script."""

import os
from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
def read_version():
    with open(os.path.join("semicart", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    raise RuntimeError("Could not read version from semicart/__init__.py")

setup(
    name="semicart",
    version=read_version(),
    author="Aydin Abedinia",
    author_email="abedinia.aydin@gmail.com",
    description="Semi-Supervised Classification and Regression Tree algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WeightedAI/semicart",
    packages=find_packages(include=["semicart", "semicart.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0",
        "pandas>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "semicart=semicart.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/WeightedAI/semicartissues",
        "Source": "https://github.com/WeightedAI/semicart",
    },
) 
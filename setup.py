""" Distribution configuration for Moprhomics
"""
from setuptools import setup
from setuptools import find_packages
import importlib

with open("README.md", "r") as fh:
    long_description = fh.read()


config = {
    "description": "MSR: measuring the relevance of neuronal firing across multiple timescales",
    "author": "Ryan Cubero",
    "url": "https://git.ist.ac.at/rcubero/MSR",
    "author_email": "cubero.ryan@gmail.com",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "install_requires": [
        "h5py>=2.8.0",
        "scipy>=0.13.3",
        "numpy>=1.8.0",
        "scikit-learn>=0.19.1",
        "cached-property>=1.5.1",
        "morphio>=2.7.1",
        "matplotlib>=3.2.0",
        "anndata>=0.7",
        "umap-learn>=0.3.10",
        "palantir>=1.0.0",
        "fa2",
    ],
    "setup_requires": ["setuptools_scm"],
    "packages": find_packages(),
    "license": "MIT",
    "scripts": [],
    "name": "MSR",
    "include_package_data": True,
    "use_scm_version": True,
    "python_requires": ">=3.7",
}


setup(**config)

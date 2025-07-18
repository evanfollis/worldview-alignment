from setuptools import setup, find_packages

setup(
    name="k_cdd_plus",
    version="1.0.0",
    description="Kernel-Coupled, Directionally-Distorted, Issue-Triggered Multi-Agent Simulator",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "numba>=0.54.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0"],
    },
)
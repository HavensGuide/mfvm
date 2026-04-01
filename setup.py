from setuptools import setup, find_packages

setup(
    name="mfvm",
    version="0.1.0",
    author="Kai Cobbs",
    author_email="",
    description=(
        "Madelung Flow Virtual Machine — a field-computing framework based on "
        "Madelung hydrodynamics. Amplitude fields (ρ, S) as a general-purpose "
        "computational substrate, GPU-native, non-destructive readout."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HavensGuide/DissipativeValue",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "matplotlib>=3.5",
        ],
        "md": [
            # For physical v-qubit / molecular dynamics use
            "mdshare>=0.3",
            "mdtraj>=1.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "madelung",
        "field computing",
        "amplitude fields",
        "quantum simulation",
        "GPU computing",
        "information geometry",
        "stochastic thermodynamics",
        "kernel methods",
    ],
)

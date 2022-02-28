# coding=utf-8

from setuptools import setup
from pathlib import Path

MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0

project_root = Path(__file__).parent

setup(
    name="fdlp_spectrogram",
    version=f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}",
    author="Samik Sadhu, Martin Kocour",
    author_email="samiksadhu@jhu.edu, ikocour@fit.vutbr.cz",
    description="Modulation Features for Automatic Speech Recognition",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=["fdlp"],
    keywords="fdlp",
    install_requires=[
        "numpy>=1.21.1",
        "scipy>=1.8.0",
        "typeguard",
        "click"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3"
    ],
    entry_points={
        'console_scripts': [
            'fdlp_spectrogram = fdlp.bin.cli:cli',
        ],
    }
)

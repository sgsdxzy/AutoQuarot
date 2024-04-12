import os
import re
from setuptools import find_packages, setup

def find_version(filepath):
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="auto_quarot",
    version=find_version("auto_quarot/__init__.py"),
    author="sgsdxzy",
    license="Apache 2.0",
    description="Auto convert transformers models to QuaRot",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["auto_quarot"]),
    install_requires=read_file("requirements.txt").strip().split("\n"),
)

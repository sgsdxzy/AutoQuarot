from setuptools import setup, find_packages


setup(
    name="auto_quarot",
    version="0.1.0",
    packages=find_packages(include=["auto_quarot"]),
    install_requires=[
        "torch",
        "transformers",
        "fast-hadamard-transform",
        "tqdm",
    ],
)
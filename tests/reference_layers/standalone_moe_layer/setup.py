from setuptools import find_packages, setup


setup(
    name="moe-standalone",
    version="0.1.0",
    description="Standalone DeepEPMOELayer extracted from ernie-core for FP8 upgrade and Sonic-MoE integration",
    packages=find_packages(include=["moe_standalone*"]),
    python_requires=">=3.8",
    install_requires=[],
)

from setuptools import setup, find_packages

setup(
    name="flower-xmkckks",
    version="0.1",
    description="A modified version of flwr",
    packages=find_packages(where="src/py"),
    package_dir={"": "src/py"},
)
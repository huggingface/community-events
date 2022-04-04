from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(name="huggan", install_requires=requirements, packages=find_packages())

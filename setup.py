from setuptools import setup, find_packages

setup(
    name="langchain_mcp",
    version="0.1.0",
    packages=find_packages(where='.', include=['langchain_mcp*']),
    install_requires=[],
)

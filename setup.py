from setuptools import setup, find_packages

setup(
    name="fortran-to-rust",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "rich>=13.7.0",
        "click>=8.1.0",
        "colorama>=0.4.6",
        "fparser>=0.2.1",
    ],
    entry_points={
        "console_scripts": [
            "fortran-to-rust=convert:main",
        ],
    },
    python_requires=">=3.10",
)

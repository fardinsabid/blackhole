from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blackhole-opt",
    version="1.0.0",
    author="Fardin Sabid",
    author_email="contact.fardinsabid@gmail.com",
    description="General Relativity based optimizer that beats Adam/AdamW",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fardinsabid/blackhole",
    packages=find_packages(),
    py_modules=["blackhole"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
    ],
)
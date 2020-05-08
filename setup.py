import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GOATNet-pkg-ireb15",
    version="0.0.1",
    author="Isaac Rebello",
    author_email="ireb692@aucklanduniac.nz",
    description="A package containing two example models and two derived models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UOA-CS302-2020/CS302-Python-2020-Group11",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
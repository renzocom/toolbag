import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toolbag",
    version="0.0.1",
    author="Renzo Comolatti",
    author_email="renzo.com@gmail.com",
    description="Toolbags with functions for signal processing, plotting, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renzocom/toolbag",
    packages=setuptools.find_packages(include=['toolbag']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
)

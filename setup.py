import setuptools

# Load the long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evrptwv2g",
    version="1.0",
    author="rariss",
    author_email="rariss@andrew.cmu.edu",
    description="Electric Vehicle Routing Problem with Time Windows and Vehicle-to-Grid (E-VRP-TW-V2G)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rariss/e-vrp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

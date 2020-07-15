import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentiment-aspects-analyzer",
    version="0.0.1",
    author="Lukasz Augustyniak",
    author_email="lukasz.augustyniak@pwr.edu.pl",
    description="The repository with source code devoted to aspect-based sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

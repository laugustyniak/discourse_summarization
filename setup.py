import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="discourse_summarization",
    version="0.0.1",
    author="Lukasz Augustyniak, Krzysztof Rajda",
    author_email="lukasz.augustyniak@pwr.edu.pl, krzysztof.rajda@pwr.edu.pl",
    description="The repository with source code devoted to aspect-based sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laugustyniak/discourse_summarization",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyppluss", # Replace with your own username
    version="0.1.5.2",
    author="Edan Rein",
    author_email="edanrein2000@gmail.com",
    description="Models for transiting exoplanets with rings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdanRein/pyPplusS",
    license="MIT",
    packages=['pyppluss'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable"
    ],
    install_requires=[
        "numpy",
        "scipy"
    ]
)

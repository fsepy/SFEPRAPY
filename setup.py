import os
from codecs import open  # To use a consistent encoding

import setuptools
import sfeprapy

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")) as f:
    long_description = f.read()

try:
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as f:
        requirements = f.read().splitlines()

setuptools.setup(
    name="SFEPRAPY",
    version=sfeprapy.__version__,
    description="Structural Fire Engineering - Probabilistic Reliability Assessment (Equivalent Time Exposure)",
    author="Ian Fu",
    author_email="fuyans@gmail.com",
    url="https://github.com/fsepy/sfeprapy",
    download_url="https://github.com/fsepy/sfeprapy/archive/master.zip",
    keywords=["fire safety", "structural fire engineering"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "sfeprapy",
        "sfeprapy.dists",
        "sfeprapy.mcs",
        "sfeprapy.mcs0",
        "sfeprapy.mcs1",
        "sfeprapy.mcs2",
        "sfeprapy.func",
    ],
    install_requires=requirements,
    include_package_data=True,
    entry_points={"console_scripts": ["sfeprapy=sfeprapy.cli.__main__:main"]},
)

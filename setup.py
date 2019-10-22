#!/usr/bin/env python

import os
from codecs import open  # To use a consistent encoding

import setuptools

import sfeprapy

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md')) as f:
    long_description = f.read()

setuptools.setup(

    name='sfeprapy',

    version=sfeprapy.__version__,

    description='Structural Fire Engineering - Probabilistic Reliability Assessment (Equivalent Time Exposure)',

    author='Ian Fu',

    author_email='fuyans@gmail.com',

    url='https://github.com/fsepy/sfeprapy',

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
        'sfeprapy',
        'sfeprapy.dat', 
        'sfeprapy.mcs0',
        'sfeprapy.mcs1',
        'sfeprapy.func',
        'sfeprapy.func.heat_transfer_1d',
        'sfeprapy.dist_fit',
        ],

    install_requires=[
        'matplotlib>=2.2.2',
        'numpy>=1.15.0',
        'pandas>=0.23.3',
        'scipy>=1.1.0',
        'seaborn>=0.9.0',
        'tqdm>=4.33',
        'xlrd>=1.2',
        'plotly>=1.12',
        'docopt>=0.6.2',
    ],

    include_package_data=True,

    entry_points={
        'console_scripts': [
            'sfeprapy=sfeprapy.cli:main'
        ],
    },
)

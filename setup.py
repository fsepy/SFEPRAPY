#!/usr/bin/env python

from distutils.core import setup

# To use a consistent encoding
from codecs import open

# Get the long description from the README file
with open('README.md') as file:
    long_description = file.read()

setup(name='sfeprapy',
      version='0.0.1',
      description='Structural fire safety engineering - robabilistic reliability assessment',
      author='Yan Fu',
      author_email='fuyans@gmail.com',
      url='https://github.com/fsepy/prapy',
      download_url="http://chardet.feedparser.org/download/python3-chardet-1.0.1.tgz",
      keywords=["fire safety", "structural fire engineering"],
      classifiers=[
          "Programming Language :: Python :: 3",
          "Development Status :: 3 - Alpha",
          "Environment :: Other Environment",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering",
      ],

      long_description=long_description,

      packages=['sfeprapy', 'sfeprapy.func', 'sfeprapy.dat', 'sfeprapy.cls'],
      )

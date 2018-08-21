#!/usr/bin/env python

from distutils.core import setup

# To use a consistent encoding
# from codecs import open
#
# Get the long description from the README file
# with open('README.md') as file:
#     long_description = file.read()

setup(name='sfeprapy',

      version='0.0.3',

      description='Structural fire safety engineering - probabilistic reliability assessment',

      author='Yan Fu',

      author_email='fuyans@gmail.com',

      url='https://github.com/fsepy/sfeprapy',

      download_url="https://github.com/fsepy/sfeprapy/archive/master.zip",

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

      long_description='Structural fire safety engineering - probabilistic reliability assessment',

      packages=['sfeprapy', 'sfeprapy.func', 'sfeprapy.dat', 'sfeprapy.cls'],

      install_requires=[
          'matplotlib>=2.2.2',
          'numpy>=1.15.0',
          'pandas>=0.23.3',
          'scipy>=1.1.0',
          'seaborn>=0.9.0',]
      )

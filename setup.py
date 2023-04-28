#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='gpmp-contrib',
      version='0.9.0',
      author='Emmanuel Vazquez',
      author_email='emmanuel.vazquez@centralesupelec.fr',
      description='GPmp contrib: the contrib GPmp package',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/gpmp-dev/gpmp-contrib',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
          "Operating System :: OS Independent",
      ],
      packages=['gpmpcontrib', 'gpmpcontrib/models', 'gpmpcontrib/optim', 'gpmpcontrib/misc', 'test'],
      license='LICENSE.txt',
      install_requires=[
             "numpy",
             "scipy>=1.8.0",
             "matplotlib",
             "gpmp>=0.9.10"
         ],
      python_requires=">=3.6",
      )

#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    author='Josue Ortega',
    author_email='caro@bcm.edu',
    url='https://github.com/josueortc/pathfinder',
    install_requires=['numpy', 'scipy','gitpython','torch=0.3.1','torchvision','scikit-image', 'matplotlib'],
)

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='subjectivity_classifier',
    version='0.1.dev0',
    packages=find_packages(),
    install_requires=[
        "Keras == 2.1.5",
        "tensorflow == 1.12.2",
        "pandas == 0.22.0",
        "nltk == 3.2.5",
        "scikit-learn == 0.19.1",
        "h5py == 2.7.1",
    ],
)

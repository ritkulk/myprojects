#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rtwik
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='writing_level_classifier',
    version='0.1.dev0',
    packages=find_packages(),
    install_requires=[
        "Keras == 2.1.5",
        "tensorflow == 1.6.0",
        "pandas == 0.22.0",
        "nltk == 3.2.5",
        "scikit-learn == 0.19.1",
        "h5py == 2.7.1",
        "pyenchant == 1.6.10"
    ],
)

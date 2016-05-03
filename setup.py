#!/usr/bin/env python
#coding=utf-8

import os
from setuptools import setup

# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "NeuronEAP",
    version = "0.1.0",
    author = "Bartosz Telenczuk, Maria Telenczuk",
    author_email = "bartosz.telenczuk@gmail.com",
    description = "Simulate extracellular fields due to action potentials",
    license = "MIT",
    url = "http://github.com/btel/neuroneap",
    packages = ['eap'],
    long_description = read('README.md'),
    classifiers = [
        "License :: OSI Approved :: MIT License"
    ],
)

#!/usr/bin/env python

from setuptools import find_packages, setup

print(f"Installing {find_packages()}")
setup(
    name="llm-annotator",
    version="0.0.1",
    description="An automated annotator for dialogue utterances.",
    author="Arthur Pan",
    author_email="s2249818@ed.ac.uk",
    packages=find_packages(),
)
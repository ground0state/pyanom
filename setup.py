# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='pyanom',
    version='0.0.1.beta',
    description='Anomaly detection package',
    long_description=readme,
    author='Masafumi Abeta',
    author_email='ground0state@gmail.com',
    install_requires=read_requirements(),
    url='https://github.com/ground0state/pyanom',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'input')),
    test_suite='tests',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
)

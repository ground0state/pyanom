import os

from setuptools import find_packages, setup

with open('README.rst') as f:
    readme = f.read()


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='pyanom',
    description='Anomaly detection package.',
    long_description=readme,
    long_description_content_type="text/x-rst",
    author='Masafumi Abeta',
    author_email='ground0state@gmail.com',
    install_requires=read_requirements(),
    url='https://github.com/ground0state/pyanom',
    license="MIT",
    packages=find_packages(exclude=('tests', 'docs', 'demo')),
    test_suite='tests',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)

# Basic setup.py structure from:
# http://stackoverflow.com/questions/16981921/relative-imports-in-python-3

from setuptools import setup, find_packages

setup(name='displ', packages=find_packages(),
        install_requires=[
            'numpy',
            'matplotlib',
            'ase',
            'numdifftools'
        ]
     )

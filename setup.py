#!/usr/bin/env python

from setuptools import (
        setup as install,
        find_packages,
        )

VERSION = '0.0.1'

install(
        name='mpi4pycuda',
        packages=['mpi4pycuda'],
        version=VERSION,
        description='mpi4py wrappers to use with pyCUDA',
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url = 'https://github.com/seba-1511/mpi4pycuda',
        download_url = 'https://github.com/seba-1511/mpi4pyucda/archive/0.0.1.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
)

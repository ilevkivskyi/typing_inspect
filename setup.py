#!/usr/bin/env python

import sys
from setuptools import setup

version = '0.9.0'
description = 'Runtime inspection utilities for typing module.'
long_description = '''
Typing Inspect
==============

The "typing_inspect" module defines experimental API for runtime
inspection of types defined in the standard "typing" module.
'''.lstrip()

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Topic :: Software Development',
]

install_requires = [
    'mypy_extensions >= 0.3.0',
    'typing_extensions >= 3.7.4',
]

setup(
    name='typing_inspect',
    version=version,
    description=description,
    long_description=long_description,
    author='Ivan Levkivskyi',
    author_email='levkivskyi@gmail.com',
    url='https://github.com/ilevkivskyi/typing_inspect',
    license='MIT',
    keywords='typing function annotations type hints hinting checking '
             'checker typehints typehinting typechecking inspect '
             'reflection introspection',
    py_modules=['typing_inspect'],
    classifiers=classifiers,
    install_requires=install_requires,
)

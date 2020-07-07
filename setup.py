#!/usr/bin/env python

# NOTE: This package must support Python 2.7 in addition to Python 3.x

import sys
from setuptools import setup

version = '0.6.0'
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
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Software Development',
]

install_requires = [
    'mypy_extensions >= 0.3.0',
    'typing >= 3.7.4;python_version<"3.5"',
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

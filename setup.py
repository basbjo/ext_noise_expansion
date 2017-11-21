#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Distutils based setup script for ext_noise_expansion.

This uses Distutils (http://python.org/sigs/distutils-sig/) the standard
python mechanism for installing packages. For the easiest installation
just type the command (you'll probably need root privileges for that):

    python setup.py install

This will install the library in the default location. For instructions on
how to customize the install procedure read the output of:

    python setup.py --help install

In addition, there are some other commands:

    python setup.py clean -> will clean all trash (*.pyc and stuff)
    python setup.py test  -> will run all doctests

To get a full list of avaiable commands, read the output of:

    python setup.py --help-commands
"""

from distutils.core import setup
from distutils.core import Command

long_description = \
'''Approximately investigate the impact of slow extrinsic fluctuations
in chemical reaction systems on the means, variances and spectrum matrix.
The series expansion[1] is based on the linear noise approximation[2,3]
and time scale separation between intrinsic and extrinsic fluctuations.

[1] Diploma thesis by Bjoern Bastian, University of Freiburg, 2013.
[2] N.G. van Kampen.  Stochastic processes in physiscs and chemistry.
    Springer-Verlag, Elsevier Science Publishers, 2 edition, 1992.
    ISBN 0-444-89439-0.
[3] J. Elf and M. Ehrenberg.  Fast evaluation of fluctuations in
    biochemical networks with the linear noise approximation.
    Genome Res, 13(11):2475--84, 2003.  doi: 10.1101/gr.1196503'''

classifiers=[
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    #"Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Biology",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.2",
    ]

class test(Command):
    """Runs all doctests
    """

    description = "run all doctests"
    user_options = []  # distutils complains if this is not here.

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        import sys
        import ext_noise_expansion
        from doctest import testmod, testfile
        testmod(ext_noise_expansion)
        testmod(ext_noise_expansion.definition_parser)
        testmod(ext_noise_expansion.definition_solvers)
        testmod(ext_noise_expansion.reaction_system)
        testmod(ext_noise_expansion.sum_evaluation)
        testmod(ext_noise_expansion.sum_generation)
        testmod(ext_noise_expansion.sum_parsing)
        testmod(ext_noise_expansion.tools_objects)
        testmod(ext_noise_expansion.tools_universal)
        testmod(ext_noise_expansion.tools_sympy)
        if sys.version_info[0] == 3:
            testfile('ext_noise_expansion.rst', module_relative=False)
        elif sys.version_info[0] == 2:
            testfile('ext_noise_expansion_python2.rst', module_relative=False)

class clean(Command):
    """Cleans *.pyc and __pycache__.
    """

    description = "remove build files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("rm -f *.pyc")
        os.system("rm -rf build")
        os.system("rm -f ext_noise_expansion/*.pyc")
        os.system("rm -rf ext_noise_expansion/__pycache__")

setup(name='ext_noise_expansion',
      version='beta',
      description="Linear noise approximation with slow extrinsic "\
              +"fluctuations.",
      long_description=long_description,
      url="https://github.com/basbjo/ext_noise_expansion",
      author="Bjoern Bastian",
      author_email="basbjo@posteo.de",
      packages=['ext_noise_expansion'],
      scripts=[],
      #platforms = ["any"],
      license="MIT",
      cmdclass={'test': test,
                'clean': clean,
                       },
      classifiers=classifiers,
      )


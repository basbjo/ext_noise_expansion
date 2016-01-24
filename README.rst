ext_noise_expansion
===================

Linear noise approximation with slow extrinsic fluctuations.

:Author:       Bjoern Bastian <basbjo at gmail dot com>
:Version:      beta
:Requirements: Python >= 2.7 / 3.x, sympy >= 0.7.3, yaml

Files:
    - README.rst
    - LICENSE
    - ext_noise_expansion.rst:  the documentation (in reStructuredText)
    - ext_noise_expansion_python2.rst: the same with Python 2 doctests
    - ext_noise_expansion.html: the documentation, converted to html
    - setup.py: distutils-setup.py and test (run doctests)
    - test_system.yaml: example reaction system definition.

  In the ext_noise_expansion directory:

    - definition_parser.py: string parser for reaction system definition
    - definition_solvers.py: solvers for stationary state evaluation
    - __init__.py: initialization of the ext_noise_expansion package
    - reaction_system.py: reaction system definition and derivatives
    - sum_evaluation.py: functions for sum evaluation
    - sum_generation.py: functions for sum generation
    - sum_parsing.py: functions for sum parsing
    - tools_objects.py: test symbol and error classes
    - tools_sympy.py: helper functions that require sympy
    - tools_sympy_doctest_python2.rst: python 2 doctests
    - tools_universal.py: general helper functions

Installation:
    - use distutils, e.g.
      ``python ./setup.py install`` or
      ``python ./setup.py install --user``
    - or simply copy the directory ext_noise_expansion
      to a directory which is found by Python's "import",
      e.g. into your project-directory

Test::

    python ./setup.py test

Issues:
    - The ``ReactionSystem`` flag ``factorize=True`` is combination with
      usage of the ``num_eval()`` method may lead to wrong results.

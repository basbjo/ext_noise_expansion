#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Linear noise approximation with slow extrinsic fluctuations.

Approximately investigate the impact of slow extrinsic fluctuations in
chemical reaction systems on the means, variances and spectrum matrix.
The series expansion[1] is based on the linear noise approximation[2,3]
and time scale separation between intrinsic and extrinsic fluctuations.

[1] Diploma thesis by Bjoern Bastian, University of Freiburg, 2013.
[2] N.G. van Kampen.  Stochastic processes in physiscs and chemistry.
    Springer-Verlag, Elsevier Science Publishers, 2 edition, 1992.
    ISBN 0-444-89439-0.
[3] J. Elf and M. Ehrenberg.  Fast evaluation of fluctuations in
    biochemical networks with the linear noise approximation.
    Genome Res, 13(11):2475--84, 2003.  doi: 10.1101/gr.1196503

For documentation see ext_noise_expansion.rst or ext_noise_expansion.html
and the module docstrings.

:Quickstart:
    System definition:
        >>> from ext_noise_expansion import ReactionSystem, simple_solve
        >>> from ext_noise_expansion import sum_evaluation as evaluate
        >>> rs = ReactionSystem.from_string(yaml_file='test_system.yaml',
        ...     factorize=True)
        >>> rs.eval_at_phis(solver=simple_solve) # stationary state
        >>> eigenvalues = rs.check_eigenvalues()
        The eigenvalues of the Jacobian A are [-k]
        for vanishing extrinsic fluctuations.

    Mean -- diploma thesis equation (3.90):
        >>> evaluate.mean(u=0, system=rs)[0]
        l/k
        >>> evaluate.mean(u=1, system=rs)[0]
        epsilon**2*l/k

    Variance of xi -- diploma thesis equation (3.91):
        >>> evaluate.variance_xi(u=0, system=rs)[0]
        l/(Omega*k)

    Variance of phis -- diploma thesis equation (3.97):
        >>> evaluate.variance_phis(u=1, system=rs)[0]
        epsilon**2*l**2/k**2

    Spectrum part from intrinsic noise -- diploma thesis equation (3.105):
        >>> evaluate.spectrum_int(u=0, system=rs)[0]
        l/(pi*Omega*(k**2 + omega**2))

    Spectrum part from extrinsic noise -- diploma thesis equation (3.110):
        >>> evaluate.spectrum_ext(u=1, system=rs)[0]
        K*epsilon**2*l**2/(pi*k**2*(K**2 + omega**2))

:Date:      2018-02-07

:Requires:  Python >= 2.7 / 3.x (recommended), sympy, yaml
:Author:    Bjoern Bastian (bjoern.bastian@uibk.ac.at)
:Copyright: Bjoern Bastian
:License:   MIT/X11-like, see __license__
"""
__version__ = "1.0"
__author__   = "Bjoern Bastian (bjoern.bastian@uibk.ac.at)"
__license__  = """Copyright (c) 2013-2018 by Bjoern Bastian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE."""

from sympy import Symbol, Matrix, pprint, solve

from .definition_parser import string_parser
from .definition_solvers import simple_solve
from .reaction_system import ReactionSystem, ReactionSystemBase
from .sum_evaluation import mean, variance_xi, variance_phis, \
        spectrum_int, spectrum_ext, partial_sums
from .tools_sympy import matsimp


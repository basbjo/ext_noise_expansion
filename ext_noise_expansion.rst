.. -*- coding: ascii -*-

===================
ext_noise_expansion
===================

:Author:    Bjoern Bastian (basbjo at gmail dot com)
:Date:      2013-

.. contents:: **Table of Contents**
.. sectnum::

-----

Description and references
==========================

Approximately investigate the impact of slow extrinsic fluctuations in chemical
reaction systems on the means, variances and spectrum matrix.  The series
expansion [1]_ is based on the linear noise approximation [2]_, [3]_ and time
scale separation between intrinsic and extrinsic fluctuations.

.. [1] Diploma thesis by Bjoern Bastian, University of Freiburg, 2013.
.. [2] N.G. van Kampen.  Stochastic processes in physiscs and chemistry.
    Springer-Verlag, Elsevier Science Publishers, 2 edition, 1992.
    ISBN 0-444-89439-0.
.. [3] J. Elf and M. Ehrenberg.  Fast evaluation of fluctuations in
    biochemical networks with the linear noise approximation.
    Genome Res, 13(11):2475--84, 2003.  `doi: 10.1101/gr.1196503`__

__ http://dx.doi.org/10.1101/gr.1196503


Quickstart
==========

The lecture of the following sections will be sufficient for basic usage.

* `System definition`_
* `Evaluation at stationary state`_
* `Numerical evaluation`_
* `The series expansions`_ (a full example)

A short example::
    >>> from ext_noise_expansion import *
    >>> rs = ReactionSystem.from_string(yaml_file='test_system.yaml')

    >>> partial_sums(2, -1, 0, # maximum u for mean/variance/spectrum
    ...     rs, solver=simple_solve) #doctest: +NORMALIZE_WHITESPACE
    {'specext_0': Matrix([[0]]),
     'specint_0': Matrix([[l/(pi*Omega*(k**2 + omega**2))]]),
     'spec_0': Matrix([[l/(pi*Omega*(k**2 + omega**2))]]),
     'mean_0': Matrix([[l/k]]),
     'mean_1': Matrix([[epsilon**2*l/k + l/k]]),
     'mean_2': Matrix([[epsilon**4*l/(2*k) + epsilon**2*l/k + l/k]])}


The ReactionSystem class
========================

System definition
-----------------
System definition with a yaml file:

 system definition::

    >>> from ext_noise_expansion import ReactionSystem
    >>> rs1 = ReactionSystem.from_string(yaml_file='test_system.yaml')
    >>> print(list(rs1.A), list(rs1.B), list(rs1.DM))
    [-k*(eta + 1)] [sqrt(k*phi*(eta + 1) + l)] [k*phi*(eta + 1) + l]

 Use ``string_parser()`` to check definition files.
 See ``test_system.yaml`` for an example definition.

System definition with a dictionary:

 system definition::

    >>> from ext_noise_expansion import ReactionSystem
    >>> data = {'concentrations': ['phi'],
    ...         'extrinsic_variables': ['eta'],
    ...         'stoichiometric_matrix': [[1, -1]],
    ...         'transition_rates': ['l', 'k*(1 + eta)*phi']}
    >>> rs2 = ReactionSystem.from_string(data)
    >>> rs2.DM == rs1.DM
    True

 etavars and etaKs are created if missing in data::

    >>> print(list(rs2.etaKs), list(rs2.etavars))
    [K1] [epsilon1**2]

Evaluation at stationary state
------------------------------
In general it will be necessary to first do a numerical evaluation.
Evaluation of the stationary state by means of a solver function::

    >>> from ext_noise_expansion import simple_solve
    >>> rs1.eval_at_phis(solver=simple_solve)
    >>> print(list(rs1.A), list(rs1.B), list(rs1.DM), list(rs1.phis))
    [-k*(eta + 1)] [sqrt(2)*sqrt(l)] [2*l] [l/(k*(eta + 1))]

External evaluation of the stationary state (for one component).
This is the method of choice for under-determined equations::

    >>> from sympy import solve
    >>> phis = solve(rs2.g[0], rs2.phi[0])
    >>> rs2.eval_at_phis(phis)
    >>> print(list(rs2.phis), rs2.g[0] == 0)
    [l/(k*(eta + 1))] True

Check if the Jacobian has eigenvalues with negative real part::

    >>> eigenvalues = rs.check_eigenvalues()
    The eigenvalues of the Jacobian A are [-k]
    for vanishing extrinsic fluctuations.

Derivatives and theta symbol
----------------------------
Taylor coefficients and theta symbol::

    >>> list(rs1.eval_symbol('A', (1,)))
    [-k]
    >>> list(rs1.eval_symbol('B', ()))
    [sqrt(2)*sqrt(l)]
    >>> list(rs1.eval_symbol('C', ((1, 1), (1, 2))))
    [l/(2*k)]
    >>> list(rs1.theta_R( ( (((1,1),(1,1)), ((1,1),(1,2))), 1) ))
    [epsilon**4/(K + k + I*omega)**2]


Numerical evaluation
--------------------
Numerical evaluation (only once or always make a copy)::

    >>> copy = rs1.copy()
    >>> rs1.num_eval({'k': 5, 'l': 3}, ifevalf=False)
    >>> print(list(rs1.A), list(rs1.B), list(rs1.DM), list(rs1.phis))
    [-5*eta - 5] [sqrt(6)] [6] [3/(5*(eta + 1))]
    >>> list(rs1.eval_symbol('C', ((1, 1), (1, 2))))
    [3/10]
    >>> list(copy.eval_symbol('C', ((1, 1), (1, 2))))
    [l/(2*k)]

    See also 'additional map_dict entry' below.

Numerical evaluation with symbol keys instead of string keys::

    >>> from sympy import symbols
    >>> k, l = symbols('k l', positive=True)
    >>> rs2.num_eval({k: 5, l:3}, map_dict=None, ifevalf=False)
    >>> rs1.A == rs2.A
    True

Detailed system definition
--------------------------
This section demonstrates some details of ReactionSystem.from_string()
and ReactionSystem.eval_at_phis()::

    >>> from sympy import Symbol, Matrix, solve
    >>> from ext_noise_expansion import ReactionSystem

Number of constituents::

    >>> N = 1 # number of components
    >>> M = 1 # number of stochastic variables
    >>> R = 2 # number of reactions

Concentrations::

    >>> phi1 = Symbol('phi1', positive=True) # concentrations
    >>> phi = Matrix([phi1])

External fluctuations::

    >>> eta1 = Symbol('eta1', positive=True) # external fluctuations
    >>> eta = Matrix([eta1])

Corresponding variances and inverse correlation times (optional,
these symbols can be created by ReactionSystem)::

    >>> epsilon1 = Symbol('epsilon1', positive=True) # std deviation
    >>> K1 = Symbol('K1', positive=True) # invers correlation time
    >>> etavars = Matrix([epsilon1**2])  # variances
    >>> etaKs   = Matrix([K1])           # inverse correlation times

Constants::

    >>> k = Symbol('k', positive=True)
    >>> l = Symbol('l', positive=True)

Stoichiometrix matrix::

    >>> S = Matrix(N, R, [ 1, -1 ])

Transition rates with fluctuations inserted as (1 + etai)::

    >>> f = Matrix(R, 1, [ l, k*(1 + eta1)*phi1 ])

Macroscopic stationary state::

    >>> phis = solve((S*f)[0], phi1)

ReactionSystem::

    >>> rs = ReactionSystem({'phi': phi, 'eta': eta, 'S': S, 'f': f})
    >>> rs.f
    Matrix([
    [                l],
    [k*phi1*(eta1 + 1)]])
    >>> rs.C
    >>> rs.eval_at_phis(phis, C_attempt=True)
    >>> list(rs.C)
    [2*l/(k*(eta1__1 + eta1__2 + 2))]

Much more detailed output
-------------------------
System definition::

    >>> rs = ReactionSystem.from_string(data, C_attempt=True,
    ...     verbose=True, pretty=False) #doctest:+ELLIPSIS
    === string_parser ===
    The chemical network consists of
         1 component[s],
         2 reactions and
         1 extrinsic stochastic variable[s].
    Concentrations of the components:
        [phi]
    Extrinsic stochastic variables:
        [eta]
    Variances (normal distribution):
        None
    Inverse correlation times:
        None
    Stoichiometric matrix:
        [1, -1]
    Macroscopic transition rates:
        [l]
        [k*phi*(eta + 1)]
    === __init__ ===
    g =
        [-k*phi*(eta + 1) + l]
    A =
        [-k*(eta + 1)]
    B =
        [sqrt(k*phi*(eta + 1) + l)]
    DM =
        [k*phi*(eta + 1) + l]
    C =
        [sqrt(...)*sqrt(...)/(k*(eta__1 + eta__2 + 2))]

Some newly created objects::

    >>> print(list(rs.phi), list(rs.eta), list(rs.S),
    ...     list(rs.etavars), list(rs.etaKs)) #doctest:+ELLIPSIS
    [phi] [eta] [1, -1] [epsilon1**2] [K1]
    >>> rs.eta_dict
    {1: eta}
    >>> rs.etai_dict
    {(1, 2): eta__2, (1, 1): eta__1}
    >>> rs.map_dict #doctest:+ELLIPSIS
    {'phi': phi, 'eta': eta, 'omega': omega, 'k': k, 'l': l, 'K1': K1, ...}

The calculation of matrix C is very inefficient (C_attempt=False
by default) and in general, only the Taylor coefficients of C can
be determined. If C can not be calculated, rs.C is set to None.

Taylor coefficients before stationary state evaluation::

    >>> list(rs.eval_symbol('C', ()))
    [(k*phi + l)/(2*k)]
    >>> list(rs.eval_symbol('B', ()))
    [sqrt(k*phi + l)]

Failing stationary state evaluation::

    >>> rs.eval_at_phis(Matrix([Symbol('phis', positive=True)]))
    WARNING: g seems not to be zero in stationary state.

After the previous evaluation we have to recreate rs::

    >>> rs = ReactionSystem.from_string(data, C_attempt=True)

    >>> rs.eval_at_phis(solver=simple_solve, verbose=True)
    === eval_at_phis ===
    phis =
        [l/(k*(eta + 1))]
    A =
        [-k*(eta + 1)]
    B =
        [sqrt(2)*sqrt(l)]
    DM =
        [2*l]
    C =
        [2*l/(k*(eta__1 + eta__2 + 2))]

Numerical evaluation with additional map_dict entry::

    >>> from sympy import Rational
    >>> rs.num_eval({'k': 5, 'l': 3, 'K1': 7, 'eps1': Rational(1,10)},
    ... map_dict={'eps1': Symbol('epsilon1', positive=True)},
    ... ifevalf=False, verbose=True)
    === num_eval ===
    phis =
        [3/(5*(eta + 1))]
    A =
        [-5*eta - 5]
    B =
        [sqrt(6)]
    DM =
        [6]
    C =
        [6/(5*(eta__1 + eta__2 + 2))]
    >>> list(rs.eval_symbol('C', ((1, 1), (1, 2))))
    [3/10]
    >>> list(rs.etavars)
    [1/100]


The series expansions
=====================
Example definitions::

    >>> from sympy import O
    >>> from ext_noise_expansion import ReactionSystem, simple_solve
    >>> from ext_noise_expansion import sum_evaluation as evaluate
    >>> rs = ReactionSystem.from_string(yaml_file='test_system.yaml')
    >>> rs.eval_at_phis(solver=simple_solve) # stationary state
    >>> eigenvalues = rs.check_eigenvalues() # negative real parts?
    The eigenvalues of the Jacobian A are [-k]
    for vanishing extrinsic fluctuations.

Partial sums of mean, variance and spectrum::

    >>> from ext_noise_expansion import partial_sums
    >>> sums = partial_sums(2, 1, 0, rs, varsimp=True)
    >>> for key in sorted(sums.keys()):
    ...     print("%s = %s" % (key, sums[key][0]))
    mean_0 = l/k
    mean_1 = epsilon**2*l/k + l/k
    mean_2 = epsilon**4*l/(2*k) + epsilon**2*l/k + l/k
    spec_0 = l/(pi*Omega*(k**2 + omega**2))
    specext_0 = 0
    specint_0 = l/(pi*Omega*(k**2 + omega**2))
    var_0 = l/(Omega*k)
    var_1 = epsilon**2*l*(Omega*l + k)/(Omega*k**2) + l/(Omega*k)
    varphis_0 = 0
    varphis_1 = epsilon**2*l**2/k**2
    varxi_0 = l/(Omega*k)
    varxi_1 = epsilon**2*l/(Omega*k) + l/(Omega*k)

    (spec = specext + specint, var = varphis + varxi)

Mean -- diploma thesis equation (3.90)::

    >>> evaluate.mean(u=0)[0]
    phis[0]
    >>> mean = evaluate.mean(u=0, system=rs)[0]
    >>> for u in range(1,3):
    ...     mean += evaluate.mean(u, rs)[0]
    >>> mean + O(rs.etavars[0]**(u+1))
    l/k + epsilon**2*l/k + epsilon**4*l/(2*k) + O(epsilon**6)

Variance of xi -- diploma thesis equation (3.91)::

    >>> evaluate.variance_xi(u=0)[0]
    C[0, 0]/Omega
    >>> var_xi = evaluate.variance_xi(u=0, system=rs)[0]
    >>> for u in range(1, 2):
    ...     var_xi += evaluate.variance_xi(u, rs)[0]
    >>> var_xi + O(rs.etavars[0]**(u+1))
    l/(Omega*k) + epsilon**2*l/(Omega*k) + O(epsilon**4)

Variance of phis -- diploma thesis equation (3.97)::

    >>> evaluate.variance_phis(u=1)[0]
    epsilon1**2*phis[1]**2
    >>> var_phis = evaluate.variance_phis(u=1, system=rs)[0]
    >>> for u in range(2, 3):
    ...     var_phis += evaluate.variance_phis(u, rs)[0]
    >>> var_phis + O(rs.etavars[0]**(u+1))
    epsilon**2*l**2/k**2 + 5*epsilon**4*l**2/(2*k**2) + O(epsilon**6)

Spectrum part from intrinsic noise -- diploma thesis equation (3.105)::

    >>> evaluate.spectrum_int(u=0)[0]
    -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
    >>> evaluate.spectrum_int(u=0, system=rs)[0] + O(rs.etavars[0])
    l/(pi*Omega*(k**2 + omega**2)) + O(epsilon**2)

    (Use the option ``together=False`` to omit the last simplification step.)

Spectrum part from extrinsic noise -- diploma thesis equation (3.110)::

    >>> evaluate.spectrum_ext(u=1)[0]
    K1*epsilon1**2*phis[1]**2/(pi*(K1**2 + omega**2))
    >>> evaluate.spectrum_ext(u=1, system=rs)[0] + O(rs.etavars[0]**2)
    K*epsilon**2*l**2/(pi*k**2*(K**2 + omega**2)) + O(epsilon**4)

Full numerical evaluation::

    >>> copy = rs.copy()
    >>> copy.num_eval({'K': 0, 'k': 5, 'l': 2, 'epsilon': 0.01})
    >>> evaluate.spectrum_int(u=0, system=copy, ifevalf=True)[0]
    0.636619772367581/(Omega*(omega**2 + 25.0))

The Jacobian must have eigenvalues with negative real part only.
This may be tested with ReactionSystem.check_eigenvalues()::

    >>> copy = rs.copy()
    >>> copy.num_eval({'K': 0, 'k': -1, 'l': 2, 'epsilon': 0.01})
    >>> copy.check_eigenvalues() #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ext_noise_expansion.tools_objects.EigenvalueError: ...

Numerical evaluation and check of eigenvalues with partial_sums()::

    >>> partial_sums(0, -1, -1, rs,
    ...         num_dict={'K': 0, 'k': 1, 'l': 2, 'epsilon': 0.01},
    ...         check_eigenvalues=True)
    {'mean_0': Matrix([[2.0]])}


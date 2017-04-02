#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Generators for sum evaluation.

Examples
========

See ext_noise_expansion.rst for usage with ReactionSystem class.

The first terms of the mean:
    >>> for u in [0, 1]:
    ...     mean(u)[0]
    phis[0]
    epsilon1**2*phis[1, 1]/2

    The symbol phis[0] denotes phis for zero eta_i.  The symbols
    phis[i,...] denote partial derivatives with respect to eta_i.

The first terms of the variance:
    >>> for u in [0, 1]:
    ...     variance_xi(u)[0]
    ...     variance_phis(u)[0] #doctest: +ELLIPSIS
    C[0, 0]/Omega
    0
    C[(1, 1), (1, 1)]*epsilon1**2/(2*Omega) + ...
    epsilon1**2*phis[1]**2

The first term of the spectrum:
    >>> spectrum_int(u=0)[0]
    -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
    >>> spectrum_ext(u=0)[0]
    0

Several partial sums:
    >>> system = ReactionSystemBase(1) # 1 extrinsic variable

    # define maximum u for mean, variance and spectrum
    >>> sums = partial_sums(1, -1, 0, system)
    >>> for key in sorted(sums.keys()):
    ...     print("%s = %s" % (key, sums[key][0]))
    mean_0 = phis[0]
    mean_1 = epsilon1**2*phis[1, 1]/2 + phis[0]
    spec_0 = -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
    specext_0 = 0
    specint_0 = -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
"""

from sympy import factorial, Matrix, eye, zeros, Wild, factor, collect, Mul
from .reaction_system import ReactionSystemBase
from .tools_sympy import diag, matsimp, def_N
from .tools_universal import merge_yield, sort_merge
from . import sum_generation as generate
from . import sum_parsing as parse

#=========================================
# evaluation of several partial sums
def partial_sums(umax_mean, umax_var, umax_spec, system, termsimp=None,
        num_dict=None, ifevalf=None, map_dict=True, solver=None, select=0,
        varsimp=False, chop_imag=None, check_eigenvalues=False,
        covariances=True, off_diagonals=False):
    """
    Calculate partial sums up to order umax. Optionally, apply
    numerical and/or stationary state evaluation.

    :Parameters:
        - `umax_mean`: maximum 'u' for partial sums of the mean
        - `umax_var`:  maximum 'u' for partial sums of the variance
        - `umax_spec`: maximum 'u' for partial sums of the spectrum
        - `system`:    instance of ReactionSystem[Base]
        - `termsimp`: function to simplify the terms of each order u
                      (optional; not applied to 'var' and 'spec')
        - `num_dict`: dictionary with substitutions, e.g. {'k': 5};
        - `ifevalf`:  if '.evalf()' is applied: True, False or None where
                      the latter implies True for non-empty 'num_dict'
        - `map_dict`: dictionary to map string num_dict-keys to symbols
                      (optional, see ReactionSystem.num_eval())
        - `solver`: solver function, phis = solver(self.g, self.phi, select)
                      or phis = solver(self.g, self.phi), e.g. sympy.solve
        - `select`: selected solution from solver (may be ambiguous)
        - `varsimp`:  simplify the partial sums of 'var'
        - `chop_imag`: Omit imaginary parts if abs(imag/real) < chop_imag
                in the calculation of the Taylor coefficients of C. If
                chop_imag is True, the default value CHOP_IMAG is used.
        - `check_eigenvalues`: raise EigenvalueError if the eigenvalues
                of the Jacobian have positive real parts
        - `covariances`: calculate full covariance matrix or diagonals only
        - `off_diagonals`: calculate full spectrum matrix or diagonals only

    :Returns: dictionary of partial sums '{<label>_<umax>: value, ...}',
        e.g. 'mean_1' is the partial sum 'mean(u=0) + mean(u=1)'

        The labels denote the following objects
        (see diploma thesis sections 3.2 and 3.3):

        ======== =========================================================
        mean      Mean(x^s)
        varphis   Var(\\phi^s)
        varxi     Var(\\xi) / \\Omega
        specint   Mean(P_i(\\omega) / \\Omega
        specext   P_e(\\omega)
        var       Var(x^s) = Var(\\phi^s) + Var(\\xi) / \\Omega
        spec      P(\\omega) = P_e(\\omega) + Mean(P_i(\\omega)) / \\Omega
        ======== =========================================================

    :Example:
        >>> sums = partial_sums(1, -1, 0, ReactionSystemBase(1))
        >>> for key in sorted(sums.keys()):
        ...     print("%s = %s" % (key, sums[key][0]))
        mean_0 = phis[0]
        mean_1 = epsilon1**2*phis[1, 1]/2 + phis[0]
        spec_0 = -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
        specext_0 = 0
        specint_0 = -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
    """
    results = {}
    copy = system.copy()
    if not termsimp:
        def termsimp(arg):
            return arg

    # numerical evaluation
    if num_dict:
        if ifevalf == None:
            ifevalf = True
        copy.num_eval(num_dict, map_dict, ifevalf=ifevalf)
    elif ifevalf:
        copy.num_eval({}, ifevalf=True)
    N = def_N(ifevalf)

    # stationary state evaluation
    if solver:
        copy.eval_at_phis(solver=solver, select=select)

    # check eigenvalues
    if check_eigenvalues:
        # raise EigenvalueError for positive real parts
        copy.check_eigenvalues(verbose=False)

    # mean
    for u in range(umax_mean+1):
        value = termsimp(N(mean(u, copy, ifevalf=ifevalf)))
        if u > 0:
            value += results['mean_'+str(u-1)]
        results.update({'mean_'+str(u): value})

    # variances
    for u in range(umax_var+1):
        valuephis = termsimp(N(variance_phis(u, copy, ifevalf=ifevalf,
                covariances=covariances)))
        valuexi = termsimp(N(variance_xi(u, copy, ifevalf=ifevalf,
            chop_imag=chop_imag, covariances=covariances)))
        both = valuephis + valuexi
        if varsimp:
            both = both.expand().applyfunc(lambda i:
                    collect(i, copy.etavars))
            if not ifevalf:
                both = both.applyfunc(factor)
        if u > 0:
            valuexi += results['varxi_'+str(u-1)]
            valuephis += results['varphis_'+str(u-1)]
            both += results['var_'+str(u-1)]
        results.update({'varxi_'+str(u): valuexi})
        results.update({'varphis_'+str(u): valuephis})
        results.update({'var_'+str(u): both})

    # spectrum
    for u in range(umax_spec+1):
        valueext = termsimp(N(spectrum_ext(u, copy, ifevalf=ifevalf,
                off_diagonals=off_diagonals)))
        valueint = termsimp(N(spectrum_int(u, copy, ifevalf=ifevalf,
                chop_imag=chop_imag, off_diagonals=off_diagonals)))
        both = valueext + valueint
        if u > 0:
            valueint += results['specint_'+str(u-1)]
            valueext += results['specext_'+str(u-1)]
            both += results['spec_'+str(u-1)]
        results.update({'specint_'+str(u): valueint})
        results.update({'specext_'+str(u): valueext})
        results.update({'spec_'+str(u): both})

    return results

#=========================================
# evaluation of the mean
def mean(u=0, system=ReactionSystemBase(1), ifevalf=False):
    """
    Returns u'th term of the expansion for the mean.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.90)

    :Example:
        >>> for u in [0, 1]:
        ...     mean(u)[0]
        phis[0]
        epsilon1**2*phis[1, 1]/2
    """
    M = system.M
    if not (isinstance(u, int) and u >= 0):
        raise ValueError("u must be a non-negative integer")
    if not system.phis:
        raise ValueError("Please evaluate at stationary state.")
    return eval_mean(parse.mean(generate.mean(u, M)), system=system,
            ifevalf=ifevalf)

#-----------------------------------------
def eval_mean(gen, system=ReactionSystemBase(1), ifevalf=False):
    """
    Evaluate and simplify the sum for the mean.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.90)

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> from ext_noise_expansion import sum_parsing
        >>> gen1 = sum_generation.mean(1, 2)
        >>> gen2 = sum_parsing.mean(gen1)
        >>> list(eval_mean(gen2, ReactionSystemBase(2)))
        [epsilon1**2*phis[1, 1]/2 + epsilon2**2*phis[2, 2]/2]
    """
    N = def_N(ifevalf)
    result = zeros(*system.phis.shape)
    for prefactor, phis_ind, epsilon_ind in gen:
        # evaluate:
        #  - factor
        #  - prefactor 1/n! for n'th derivative
        #  - partial derivatives
        #  - product of squared epsilon_i
        result += N(prefactor/factorial(len(phis_ind))
                * system.eval_symbol('phis', phis_ind)
                * system.eval_variances(epsilon_ind))
    return result

#=========================================
# evaluation of the variance matrix
def variance_xi(u=0, system=ReactionSystemBase(1),
        covariances=True, ifevalf=False, chop_imag=None):
    """
    Returns u'th term of the expansion for the variance_xi.
    If not covariances only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    Omit imaginary parts if abs(imag/real) < chop_imag in the
    calculation of the Taylor coefficients of C. If chop_imag
    is True, the default value CHOP_IMAG is used.

    :Definition: diploma thesis equation (3.91); as opposed to the
        thesis, variance_xi is divided by the system size Omega

    :Example:
        >>> for u in [0, 1]:
        ...     variance_xi(u)[0] #doctest: +NORMALIZE_WHITESPACE
        C[0, 0]/Omega
        C[(1, 1), (1, 1)]*epsilon1**2/(2*Omega) +
        C[(1, 1), (1, 2)]*epsilon1**2/Omega +
        C[(1, 2), (1, 2)]*epsilon1**2/(2*Omega)
    """
    M = system.M
    if not (isinstance(u, int) and u >= 0):
        raise ValueError("u must be a non-negative integer")
    if not system.phis:
        raise ValueError("Please evaluate at stationary state.")
    return eval_variance_xi(parse.variance_xi(generate.variance_xi(u, M)),
            system=system, covariances=covariances,
            ifevalf=ifevalf, chop_imag=chop_imag)

#-----------------------------------------
def eval_variance_xi(gen, system=ReactionSystemBase(1),
        covariances=True, ifevalf=False, chop_imag=None):
    """
    Evaluate and simplify the sum for the variance_xi.
    If not covariances only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    Omit imaginary parts if abs(imag/real) < chop_imag in the
    calculation of the Taylor coefficients of C. If chop_imag
    is True, the default value CHOP_IMAG is used.

    :Definition: diploma thesis equation (3.91); as opposed to the
        thesis, variance_xi is divided by the system size Omega

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> from ext_noise_expansion import sum_parsing
        >>> gen1 = sum_generation.variance_xi(1, 1)
        >>> gen2 = sum_parsing.variance_xi(gen1)
        >>> list(eval_variance_xi(gen2,
        ...     ReactionSystemBase(1))) #doctest: +NORMALIZE_WHITESPACE
        [C[(1, 1), (1, 1)]*epsilon1**2/(2*Omega) +
         C[(1, 1), (1, 2)]*epsilon1**2/Omega +
         C[(1, 2), (1, 2)]*epsilon1**2/(2*Omega)]
    """
    N = def_N(ifevalf)
    if covariances:
        result = zeros(*system.A.shape)
    else:
        result = zeros(*system.phis.shape)
    for prefactor, C_ind, epsilon_ind in gen:
        # evaluate:
        #  - factor
        #  - prefactor 1/n! for n'th derivative
        #  - partial derivatives
        #  - product of squared epsilon_i
        part = (prefactor/system.Omega/factorial(len(C_ind))
                * system.eval_symbol('C', C_ind, chop_imag=chop_imag)
                * system.eval_variances(epsilon_ind))
        if not covariances:
            part = Matrix([part[i, i] for i in range(part.rows)])
        result += N(part)
    return result

#-----------------------------------------
def variance_phis(u=0, system=ReactionSystemBase(1),
        covariances=True, ifevalf=False):
    """
    Returns u'th term of the expansion for the variance_phis.
    If not covariances only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.97)

    :Example:
        >>> for u in [1, 2]:
        ...     variance_phis(u)[0] #doctest: +NORMALIZE_WHITESPACE
        epsilon1**2*phis[1]**2
        epsilon1**4*phis[1, 1, 1]*phis[1] + epsilon1**4*phis[1, 1]**2/2 +
        3*epsilon1**4*phis[1, 1]*phis[1] + epsilon1**4*phis[1]**2/2
    """
    M = system.M
    if not (isinstance(u, int) and u >= 0):
        raise ValueError("u must be a non-negative integer")
    if not system.phis:
        raise ValueError("Please evaluate at stationary state.")
    return eval_variance_phis(parse.variance_phis(generate.variance_phis(u, M)),
            system=system, covariances=covariances, ifevalf=ifevalf)

#-----------------------------------------
def eval_variance_phis(gen, system=ReactionSystemBase(1),
        covariances=True, ifevalf=False):
    """
    Evaluate and simplify the sum for the variance_phis.
    If not covariances only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.97)

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> from ext_noise_expansion import sum_parsing
        >>> gen1 = sum_generation.variance_phis(1, 1)
        >>> gen2 = sum_parsing.variance_phis(gen1)
        >>> list(eval_variance_phis(gen2,
        ...     ReactionSystemBase(1)))
        [epsilon1**2*phis[1]**2]
    """
    N = def_N(ifevalf)
    if covariances:
        result = zeros(*system.A.shape)
    else:
        result = zeros(*system.phis.shape)
    for prefactor, r1_ind, r2_ind, epsilon_ind in gen:
        # evaluate:
        #  - factor
        #  - prefactor 1/n! for n'th derivative
        #  - partial derivatives
        #  - product of squared epsilon_i
        part = (prefactor/factorial(len(r1_ind))/factorial(len(r2_ind))
                * system.eval_symbol('phis', r1_ind)
                * system.eval_symbol('phis', r2_ind).T
                * system.eval_variances(epsilon_ind))
        if not covariances:
            part = Matrix([part[i, i] for i in range(part.rows)])
        result += N(part)
    return result

#=========================================
# evaluation of the spectrum matrix
def spectrum_int(u=0, system=ReactionSystemBase(1), together=True,
        off_diagonals=False, ifevalf=False, chop_imag=None):
    """
    Returns u'th term of the expansion for the spectrum_int.

    :Parameters:
        - `u`: order of the expansion
        - `system`: instance of ReactionSystem[Base]
        - `together`: expand terms with identical denominators except for
                an exponent and merge to a common denominator
        - `off_diagonals`: calculate full spectrum matrix or diagonals only
        - `ifevalf`: if .evalf() is called in the end
        - `chop_imag`: omit imaginary parts if abs(imag/real) < chop_imag
                in the calculation of the Taylor coefficients of C; if
                chop_imag is True, the default value CHOP_IMAG is used.

    :Definition: diploma thesis equation (3.105); as opposed to the
        thesis, spectrum_int is divided by the system size Omega

    :Example:
        >>> spectrum_int(0)[0]
        -A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))
    """
    M = system.M
    if not (isinstance(u, int) and u >= 0):
        raise ValueError("u must be a non-negative integer")
    if not system.phis:
        raise ValueError("Please evaluate at stationary state.")
    return eval_spectrum_int(parse.spectrum_R(generate.spectrum_R(u, M)),
            system=system, together=together, off_diagonals=off_diagonals,
            ifevalf=ifevalf, chop_imag=chop_imag)

#-----------------------------------------
def eval_spectrum_int(gen, system=ReactionSystemBase(1), together=True,
        off_diagonals=False, ifevalf=False, chop_imag=None):
    """
    Evaluate and simplify the sum for spectrum_int.

    :Parameters:
        - `u`: order of the expansion
        - `system`: instance of ReactionSystem[Base]
        - `together`: expand terms with identical denominators except for
                an exponent and merge to a common denominator
        - `off_diagonals`: calculate full spectrum matrix or diagonals only
        - `ifevalf`: if .evalf() is called in the end
        - `chop_imag`: omit imaginary parts if abs(imag/real) < chop_imag
                in the calculation of the Taylor coefficients of C; if
                chop_imag is True, the default value CHOP_IMAG is used.

    :Definition: diploma thesis equation (3.105); as opposed to the
        thesis, spectrum_int is divided by the system size Omega

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> from ext_noise_expansion import sum_parsing
        >>> gen1 = sum_generation.spectrum_R(0, 1)
        >>> gen2 = sum_parsing.spectrum_R(gen1)
        >>> list(eval_spectrum_int(gen2,
        ...     ReactionSystemBase(1)))
        [-A[0]*C[0, 0]/(pi*Omega*(A[0]**2 + omega**2))]
    """
    N = def_N(ifevalf)
    def first_eval(gen):
        """evaluate except for theta symbols"""
        # evaluate:
        #  - factor
        #  - partial derivatives
        #  - prefactor 1/n! for n'th derivative
        for prefactor, theta_ind, A_indices, C_ind in gen:
            part = prefactor * eye(system.A.rows)
            for A_ind in A_indices:
                part *= system.eval_symbol('A', A_ind)/factorial(len(A_ind))
            part *= system.eval_symbol('C', C_ind,
                    chop_imag=chop_imag)/factorial(len(C_ind))
            part = matsimp(part, system.omega, factorize=system.factorize)
            # append None which will be omitted by sort_merge
            yield [ part, theta_ind, None]

    def second_eval(gen):
        """evaluate theta symbols and add complex conjugate"""
        # evaluate:
        #  - sort and sum up terms with identical theta_ind
        #  - evaluate theta_ind
        #  - add complex conjugate and simplify
        #  - divide by 2*pi
        for part, theta_ind in sort_merge(merge_yield(gen)):
            part = system.theta_R(theta_ind) * part
            # separate diagonal elements
            diagonal = [part[i, i] for i in range(part.rows)]
            for i in range(part.rows):
                part[i, i] = 0
            if off_diagonals:
                # add complex conjugate transpose
                part += part.transpose().conjugate()
                part /= (2 * system.pi * system.Omega)
                part = matsimp(part, system.omega, factorize=system.factorize)
            # the same for the diagonal elements (omit small imaginary parts)
            for i, item in enumerate(diagonal):
                terms = []
                for term in item.as_coeff_add()[1]:
                    term += term.conjugate()
                    term /= (2 * system.pi * system.Omega)
                    # together and numer/denom simplification:
                    term = matsimp(Matrix([term]), chop_imag=True,
                            factorize=system.factorize)[0]
                    terms.append(term)
                diagonal[i] = sum(terms)
            # put everything together and simplify
            if off_diagonals:
                part += diag(diagonal)
            else:
                part = Matrix(diagonal)
            if together:
                # sort key for third_eval from denominator
                third_key = part[0].as_numer_denom()[1]
                if isinstance(third_key, Mul):
                    p = Wild("p")
                    q = Wild("q")
                    r = Wild("r")
                    # try to ignore an exponent in the denominator
                    third_key = third_key.match(p*q**r)[q]
                third_key = str(third_key)
            else:
                third_key = None
            # append None which will be omitted by sort_merge
            yield [ part, third_key, None ]

    def third_eval(gen):
        """optional simplification (expand and merge some fractions)"""
        if together:
            # evaluate:
            #  - sort and sum up terms with denominators that
            #    are identical except for a power
            for part, _ in sort_merge(merge_yield(gen)):
                # third_key was omitted as _
                yield N(matsimp(part, system.omega, factorize=system.factorize))
        else:
            # omit third evaluation
            for part, _, _ in gen:
                yield N(part)

    # shape of the result
    if off_diagonals:
        result = zeros(*system.A.shape)
    else:
        result = zeros(*system.phis.shape)
    # sum up terms
    for part in third_eval(second_eval(first_eval(gen))):
        result += part
    return result

#-----------------------------------------
def spectrum_ext(u=0, system=ReactionSystemBase(1),
        off_diagonals=False, ifevalf=False):
    """
    Returns u'th term of the expansion for the spectrum_ext.
    If not off_diagonals only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.110)

    :Example:
        >>> spectrum_ext(1)[0]
        K1*epsilon1**2*phis[1]**2/(pi*(K1**2 + omega**2))
    """
    M = system.M
    if not (isinstance(u, int) and u >= 0):
        raise ValueError("u must be a non-negative integer")
    if not system.phis:
        raise ValueError("Please evaluate at stationary state.")
    return eval_spectrum_ext(parse.spectrum_ext(generate.spectrum_ext(u, M)),
            system=system, off_diagonals=off_diagonals, ifevalf=ifevalf)

#-----------------------------------------
def eval_spectrum_ext(gen, system=ReactionSystemBase(1),
        off_diagonals=False, ifevalf=False):
    """
    Evaluate and simplify the sum for spectrum_ext.
    If not off_diagonals only the diagonals are calculated.
    Call .evalf() in the end if ifevalf is true.

    :Definition: diploma thesis equation (3.110)

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> from ext_noise_expansion import sum_parsing
        >>> gen1 = sum_generation.spectrum_ext(1, 1)
        >>> gen2 = sum_parsing.spectrum_ext(gen1)
        >>> list(eval_spectrum_ext(gen2,
        ...     ReactionSystemBase(1)))
        [K1*epsilon1**2*phis[1]**2/(pi*(K1**2 + omega**2))]
    """
    N = def_N(ifevalf)
    def first_eval(gen):
        """evaluate except for Theta symbols"""
        # evaluate:
        #  - factor
        #  - prefactor 1/n! for n'th derivative
        #  - partial derivatives
        for prefactor, r1_ind, r2_ind, Theta_ind in gen:
            part = prefactor * eye(system.A.rows)
            part /= factorial(len(r1_ind))*factorial(len(r2_ind))
            part *= system.eval_symbol('phis', r1_ind)
            part *= system.eval_symbol('phis', r2_ind).T
            if not off_diagonals:
                part = Matrix([part[i, i] for i in range(part.rows)])
            part = matsimp(part, factorize=system.factorize)
            # append None which will be omitted by sort_merge
            yield [ part, Theta_ind, None]
    def second_eval(gen):
        """evaluate Theta symbols"""
        # evaluate:
        #  - sort and sum up terms with identical Theta_ind
        #  - evaluate Theta_ind
        for part, Theta_ind in sort_merge(merge_yield(gen)):
            part *= system.theta_P(Theta_ind)
            yield part
    # shape of the result
    if off_diagonals:
        result = zeros(*system.A.shape)
    else:
        result = zeros(*system.phis.shape)
    # sum up terms
    for part in second_eval(first_eval(gen)):
        result += N(part)
    return result


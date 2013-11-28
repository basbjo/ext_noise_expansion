.. -*- coding: ascii -*-

====================================
Python 2 doctests for tools_sympy.py
====================================

diag(vec):

    Return a list as diagonal matrix.

    :Returns: MutableMatrix

    :Example::
        >>> from ext_noise_expansion.tools_sympy import diag
        >>> v = [1, 2, 3]
        >>> diag(v)
        [1, 0, 0]
        [0, 2, 0]
        [0, 0, 3]

deep_factor(i):

    Temporarily subtract 'omega**2' and factorize in all bases.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import deep_factor
        >>> from sympy.abc import k, omega
        >>> deep_factor((k**2 + 2*k + 1 + omega**2) ** 2 * (k + 1), omega)
        (k + 1)*(omega**2 + (k + 1)**2)**2

matsimp(m, deep=False, real_numer_denom=False):

    Simplify all entries of a MutableMatrix m:
      - common denominator
      - factorization in numerator and denominator

    :Parameters:
        - `deep`: if 'True', function 'deep_factor()' is applied
        - `real_numer_denom`: omit imaginary part in the denominator

lyapunov_equation(A, B):

    Returns Lyapunov matrix L such that
    A * L + L * A.T + B = 0.

    A and B must be square matrices.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import lyapunov_equation
        >>> from sympy import Matrix
        >>> A = Matrix(2,2,[1, 2, 0, 3])
        >>> B = Matrix(2,2,[1, 2, 2, 3])
        >>> L = lyapunov_equation(A, B)
        >>> L
        [   0, -1/4]
        [-1/4, -1/2]
        >>> A * L + L * A.T + B
        [0, 0]
        [0, 0]

lyapunov_equation_2(A1, A2, B):

    Returns Lyapunov matrix L such that
    A1 * L + L * A2.T + B = 0.

    A and B must be square matrices.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import lyapunov_equation_2
        >>> from sympy import Matrix
        >>> A1 = A2 = Matrix(2,2,[1, 2, 0, 3])
        >>> B = Matrix(2,2,[1, 2, 2, 3])
        >>> L = lyapunov_equation_2(A1, A2, B)
        >>> L
        [   0, -1/4]
        [-1/4, -1/2]
        >>> A1 * L + L * A2.T + B
        [0, 0]
        [0, 0]

derivative(func, arg_dict, keys, memo=None):

    Higher dimension derivatives of differentiable functions.

    If a dictionary is provided as 'memo', the coefficients are calculated
    iteratively and remembered.  Make sure, that the dictionary is always
    used with the same function and argument dictionary (no checks!).

    :Example::
        >>> from ext_noise_expansion.tools_sympy import derivative
        >>> from sympy.abc import x,y
        >>> f = x * y ** 3
        >>> d = {}
        >>> derivative(f, {0: x, 1: y}, (1, 0, 1), d)
        6*y
        >>> d
        {(0, 1): 3*y**2, (0, 1, 1): 6*y, (): x*y**3, (0,): y**3}

taylor_coeff(func, arg_dict, keys, memo=None):

    Higher dimension Taylor coefficients of differentiable functions.

    If a dictionary is provided as 'memo', the coefficients are calculated
    iteratively and remembered.  Make sure, that the dictionary is always
    used with the same function and argument dictionary (no checks!).

    :Example::
        >>> from ext_noise_expansion.tools_sympy import taylor_coeff
        >>> from sympy.abc import x,y
        >>> f = 3 * x * y ** 2 + 5 * x
        >>> d = {}
        >>> taylor_coeff(f, {0: x, 1: y}, (1, 0, 1), d)
        6
        >>> d #doctest: +ELLIPSIS
        {'derivatives': {...}, (0, 1, 1): 6}

multi_indices_k(q, M):

    Returns a list of multi-indices r = (r_1, ..., r_q)
    such that all r_i \\in {1, ..., M}.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import multi_indices_k
        >>> multi_indices_k(2, 2)
        [[1, 1], [1, 2], [2, 1], [2, 2]]

multi_indices_l(r, M):

    Returns a list of multi-indices l = ((l_1, m_1), ..., (l_r, m_r))
    such that all l_i \\in {1, ..., M} and m_i \\in {1, 2}.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import multi_indices_l
        >>> multi_indices_l(2, 1) #doctest: +ELLIPSIS
        [[(1, 1), (1, 1)], [(1, 1), (1, 2)], [...], [(1, 2), (1, 2)]]

inner_sum_indices(u, n):

    Returns a list of partially ordered index pair tuples
    [m, [(i_1, j_1), ..., (i_u, j_u)]] such that the pairs (i_l, j_l)
    satisfy i_l < j_l and the union of all indices is {1, ..., n}.
    m is the number of possible permutations of the index pairs.

    :Example::
        >>> from ext_noise_expansion.tools_sympy import inner_sum_indices
        >>> inner_sum_indices(2, 4) #doctest: +ELLIPSIS
        [[2, [(1, 3), (2, 4)]], ..., [2, [(1, 4), (2, 3)]]]


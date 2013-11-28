#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Helper functions specific for symbolic python.
"""

from sympy import Matrix, eye, Number, pprint
from sympy.core.add import Add
from sympy.ntheory.multinomial import multinomial_coefficients

CHOP_IMAG = 1e-12

#=========================================
# matrices

def diag(vec):
    """
    Return a list as diagonal matrix.

    :Returns: MutableMatrix

    :Example:
        >>> v = [1, 2, 3]
        >>> diag(v)
        Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]])
    """
    vec = list(vec)
    rows = len(vec)
    return Matrix(rows, rows, [0**(abs(i-j))*vec[i]
        for i in range(rows) for j in range(rows)])

#-----------------------------------------
def deep_factor(expr, omega):
    """
    Temporarily subtract omega**2 and factorize in all bases.

    :Example:
        >>> from sympy.abc import k, omega
        >>> deep_factor((k**2 + 2*k + 1 + omega**2)**2 * (k + 1), omega)
        (k + 1)*(omega**2 + (k + 1)**2)**2
    """
    expr = expr.as_powers_dict()
    for base, exponent in expr.items():
        if isinstance(base, Add):
            new_base = (base - omega**2).factor() + omega**2
            expr.pop(base)
            expr.update({new_base: exponent})
    new_expr = None
    for base, exponent in expr.items():
        if not new_expr:
            new_expr = base**exponent
        else:
            new_expr *= base**exponent
    return new_expr

#-----------------------------------------
def matsimp(matrix, omega=None, chop_imag=False):
    """
    Simplify all entries of a MutableMatrix m:
      - common denominator
      - factorization in numerator and denominator

    If a Symbol is given for omega, the function deep_factor() is
    called with omega as second argument.  If chop_imag, imaginary
    parts are omitted in numerator and common denominator.
    """
    def entrysimp(entry):
        """simplify each entry"""
        entry = entry.together()
        numer, denom = entry.as_numer_denom()
        # expansion
        numer = numer.expand()
        denom = denom.expand()
        # omit imaginary parts
        if chop_imag:
            numer = numer.as_real_imag()[0]
            denom = denom.as_real_imag()[0]
        numer = numer.factor()
        # factorization
        denom = denom.factor()
        if omega:
            denom = deep_factor(denom, omega)
        # avoid [1|-1]*(...) in the numerator
        numer = [numer, numer.as_coeff_Mul()]
        if isinstance(numer[1][0], Number):
            numer = numer[1][0] * numer[1][1]
        else:
            numer = numer[0]
        denom = [denom, denom.as_coeff_Mul()]
        if denom[1][0] == 1:
            denom = denom[1][1]
        else:
            denom = denom[0]
        return numer/denom
    return matrix.applyfunc(entrysimp)

#-----------------------------------------
def lyapunov_equation(A, B, chop_imag=None):
    """
    Returns Lyapunov matrix L such that
    A * L + L * A.T + B = 0.

    A and B must be square matrices.

    Omit imaginary parts if abs(imag/real) < chop_imag in the end.
    If chop_imag is True, the default value CHOP_IMAG is used.

    :Example:
        >>> from sympy import Matrix
        >>> A = Matrix(2,2,[1, 2, 0, 3])
        >>> B = Matrix(2,2,[1, 2, 2, 3])
        >>> L = lyapunov_equation(A, B)
        >>> L
        Matrix([
        [   0, -1/4],
        [-1/4, -1/2]])
        >>> A * L + L * A.T + B
        Matrix([
        [0, 0],
        [0, 0]])
    """
    if not A.is_square:
        raise ValueError("Matrix must be square.")
    if not B.is_square:
        raise ValueError("Matrix must be square.")
    if not A.rows == B.rows:
        raise ValueError("Matrices must be of same size.")
    if A.is_diagonal():
        a = A
        P = eye(A.rows)
    else:
        P, a = A.diagonalize()
    b = P.inv() * B * P.T.inv()
    if chop_imag:
        b = b.evalf()
    l = Matrix(b.rows, b.rows, lambda i, j:
            -b[i, j]/(a[i, i] + a[j, j]))
    if chop_imag:
        l = l.evalf()
    L = P * l * P.T
    if chop_imag:
        L = L.evalf()
    if chop_imag == True:
        chop_imag = CHOP_IMAG
    if chop_imag:
        for i, number in enumerate(L):
            real, imag = number.as_real_imag()
            if abs(imag/real) < chop_imag:
                L[i] = real
    return L

#-----------------------------------------
def lyapunov_equation_2(A1, A2, B):
    """
    Returns Lyapunov matrix L such that
    A1 * L + L * A2.T + B = 0.

    A and B must be square matrices.

    :Example:
        >>> from sympy import Matrix
        >>> A1 = A2 = Matrix(2,2,[1, 2, 0, 3])
        >>> B = Matrix(2,2,[1, 2, 2, 3])
        >>> L = lyapunov_equation_2(A1, A2, B)
        >>> L
        Matrix([
        [   0, -1/4],
        [-1/4, -1/2]])
        >>> A1 * L + L * A2.T + B
        Matrix([
        [0, 0],
        [0, 0]])
    """
    if not A1.is_square:
        raise ValueError("Matrix must be square.")
    if not A2.is_square:
        raise ValueError("Matrix must be square.")
    if not B.is_square:
        raise ValueError("Matrix must be square.")
    if not A1.rows == B.rows and A2.rows == B.rows:
        raise ValueError("Matrices must be of same size.")
    P1, a1 = A1.diagonalize()
    P2, a2 = A2.diagonalize()
    b = P1.inv() * B * P2.T.inv()
    l = Matrix(b.rows, b.rows, lambda i, j:
            -b[i, j]/(a1[i, i] + a2[j, j]))
    return P1 * l * P2.T

#=========================================
# derivatives
def derivative(func, arg_dict, keys, memo=None):
    """
    Higher dimension derivatives of differentiable functions.

    If a dictionary is provided as memo, the coefficients are calculated
    iteratively and remembered.  Make sure, that the dictionary is always
    used with the same function and argument dictionary (no checks!).

    :Example:
        >>> from sympy.abc import x,y
        >>> f = x * y ** 3
        >>> d = {}
        >>> derivative(f, {0: x, 1: y}, (1, 0, 1), d)
        6*y
        >>> d
        {(0, 1): 3*y**2, (0, 1, 1): 6*y, (): x*y**3, (0,): y**3}
    """
    keys = tuple(sorted(keys))
    if memo == None:
        # calculate derivative and return
        for key in list(keys):
            func = func.diff(arg_dict[key])
        return func
    # use memoization dictionary
    elif keys in memo:
        return memo[keys]
    elif keys == ():
        memo.update({keys: func})
        return func
    else:
        subkeys = list(keys)
        key = subkeys.pop(-1)
        subkeys = tuple(subkeys)
        func = derivative(func, arg_dict, subkeys, memo)
        func = func.diff(arg_dict[key])
        memo.update({keys: func})
        return func

#-----------------------------------------
def taylor_coeff(func, arg_dict, keys, memo=None):
    """
    Higher dimension Taylor coefficients of differentiable functions.

    If a dictionary is provided as memo, the coefficients are calculated
    iteratively and remembered.  Make sure, that the dictionary is always
    used with the same function and argument dictionary (no checks!).

    :Example:
        >>> from sympy.abc import x,y
        >>> f = 3 * x * y ** 2 + 5 * x
        >>> d = {}
        >>> taylor_coeff(f, {0: x, 1: y}, (1, 0, 1), d)
        6
        >>> d #doctest: +ELLIPSIS
        {'derivatives': {...}, (0, 1, 1): 6}
    """
    keys = tuple(sorted(keys))
    if memo == None:
        # evaluate derivative at 0 and return
        func = derivative(func, arg_dict, keys)
        for key in keys:
            func = func.subs({arg_dict[key]: 0})
        return func
    elif keys in memo:
        return memo[keys]
    else:
        if not "derivatives" in memo:
            memo["derivatives"] = {}
        func = derivative(func, arg_dict, keys, memo["derivatives"])
        for v in arg_dict.values():
            func = func.subs({v: 0})
        memo.update({keys: func})
        return func

#=========================================
# indices

_MEMO_MULTI_INDICES_K = {}
def multi_indices_k(q, M):
    """
    Returns a list of multi-indices r = (r_1, ..., r_q)
    such that all r_i \\in {1, ..., M}.

    :Example:
        >>> multi_indices_k(2, 2)
        [[1, 1], [1, 2], [2, 1], [2, 2]]
    """
    if (q, M) in _MEMO_MULTI_INDICES_K:
        # if it was already calculated
        return _MEMO_MULTI_INDICES_K[(q, M)]
    # new calculation (recursion)
    if q == 0:
        new_k = [[]]
    else:
        new_k = [ [q1] + prev_k
            for q1 in range(1, M+1)
                for prev_k in multi_indices_k(q-1, M) ]
    # remember new result
    _MEMO_MULTI_INDICES_K[(q, M)] = new_k
    return  new_k

#-----------------------------------------
def multi_indices_l(r, M):
    """
    Returns a list of multi-indices l = ((l_1, m_1), ..., (l_r, m_r))
    such that all l_i \\in {1, ..., M} and m_i \\in {1, 2}.

    :Example:
        >>> multi_indices_l(2, 1) #doctest: +ELLIPSIS
        [[(1, 1), (1, 1)], [(1, 1), (1, 2)], [...], [(1, 2), (1, 2)]]
    """
    if r == 0:
        return [[]]
    multi_l = []
    multi_k = multi_indices_k(r, M)
    m_tuples = [[1], [2]]
    for i in range(r-1):
        new_tuples = []
        for item in m_tuples:
            new_tuples.append(item + [1])
            new_tuples.append(item + [2])
        m_tuples = new_tuples
    # all l tuples
    for cur_k in multi_k[:]:
        # all m tuples
        for m_tuple in m_tuples:
            # merge l and m tuples
            new_l = []
            for i, l in enumerate(cur_k):
                new_l.append((l, m_tuple[i]))
            multi_l.append(new_l)
    return multi_l

#-----------------------------------------
_MEMO_INNER_SUM_INDICES = {}
def inner_sum_indices(u, n):
    """
    Returns a list of partially ordered index pair tuples
    [m, [(i_1, j_1), ..., (i_u, j_u)]] such that the pairs (i_l, j_l)
    satisfy i_l < j_l and the union of all indices is {1, ..., n}.
    m is the number of possible permutations of the index pairs.

    :Example:
        >>> inner_sum_indices(2, 4) #doctest: +ELLIPSIS
        [[2, [(1, 3), (2, 4)]], ..., [2, [(1, 4), (2, 3)]]]
    """
    if n < 2:
        if u == 0 and n == 0:
            # only empty set possible
            return [[1, []]]
        else:
            # no possible set of indices in this case
            return []
    if (u, n) in _MEMO_INNER_SUM_INDICES:
        # if it was already calculated
        return _MEMO_INNER_SUM_INDICES[(u, n)]
    # new calculation
    base_ind = range(1, n+1)
    base_pairs = []
    for i in base_ind:
        # all possible ordered index pairs
        for r in base_ind[i:]:
            base_pairs.append((i, r))
    l = []
    for choice, m in multinomial_coefficients(len(base_pairs), u).items():
        # all possible partially ordered selections of index pairs
        test_ind = []
        for i, times in enumerate(choice):
            if times > 0:
                test_ind += (times * [base_pairs[i]])
        i_list = []
        for i in test_ind:
            i_list += i
        if set(i_list) == set(base_ind):
            # only selections that contain all elements of base_ind
            l.append([m, test_ind])
    # remember result
    _MEMO_INNER_SUM_INDICES[(u, n)] = l
    return l

#=========================================
# misc
#-----------------------------------------
def def_nprint(pretty, indent=6):
    """
    Define a print function using pprint if pretty or print if not.
    If print is used, insert indentation of indent space characters.

    :Returns: print function
    """
    if pretty:
        def nprint(i):
            """pretty print"""
            pprint(i)
            print('')
    else:
        def nprint(i):
            """default print"""
            if i:
                if isinstance(i, Matrix):
                    i = i.tolist()
                for line in i:
                    print(" "*indent + str(line))
            elif i == None:
                print(" "*indent + "None")
    return nprint

def def_N(ifevalf):
    """
    Define a function that calls .evalf() if ifevalf.
    """
    if ifevalf:
        def N(arg):
            """call .evalf()"""
            return arg.evalf()
    else:
        def N(arg):
            """do not call .evalf()"""
            return arg
    return N


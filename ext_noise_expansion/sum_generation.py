#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Generators for sum generation.
"""

from sympy import Rational, factorial
from .tools_universal import multi_indices, multi_indices_kn_l
from .tools_sympy import inner_sum_indices, multi_indices_k, multi_indices_l
DEBUG = False

#-----------------------------------------
def mean(u, M=1):
    """
    Generate u'th order terms for the mean.
    M is the number of external variables.

    :Definition: diploma thesis equation (3.90)

    :Yields: [ prefactor, multi-index_k, Delta_index_pairs, (u, q) ]

    :Example:
        >>> next(mean(2, 1))
        [1/2, [1, 1], [(1, 2), (1, 2)], (2, 2)]
    """
    if DEBUG:
        print("u = %r, M = %r" % (u, M))
    # second sum q = 0, ..., 2u
    for q in range(2*u + 1):
        if DEBUG:
            print("  q = %r" % q)
        # third sum #r = q
        for r in multi_indices_k(q, M):
            if DEBUG:
                print("    r = %r" % r)
            # fourth sum over index pairs
            for factor, indices in inner_sum_indices(u, q):
                factor = Rational(factor, factorial(u))
                final = [ factor, r, indices, (u, q) ]
                if DEBUG:
                    print("\t\t\t%r" % final)
                yield final

#-----------------------------------------
def variance_xi(u, M=1):
    """
    Generate u'th order terms for the variance of xi.
    M is the number of external variables.

    :Definition: diploma thesis equation (3.91)

    :Yields: [ prefactor, multi-index_l, Delta_index_pairs, (u, r) ]

    :Example:
        >>> next(variance_xi(2, 1))
        [1/2, [(1, 1), (1, 1)], [(1, 2), (1, 2)], (2, 2)]
    """
    if DEBUG:
        print("u = %r, M = %r" % (u, M))
    # second sum r = 0, ..., 2u
    for r in range(2*u + 1):
        if DEBUG:
            print("  r = %r" % r)
        # third sum #l = r
        for l in multi_indices_l(r, M):
            if DEBUG:
                print("    l = %r" % l)
            # fourth sum over index pairs
            for factor, indices in inner_sum_indices(u, r):
                factor = Rational(factor, factorial(u))
                final = [ factor, l, indices, (u, r) ]
                if DEBUG:
                    print("\t\t\t%r" % final)
                yield final

#-----------------------------------------
def variance_phis(u, M=1):
    """
    Generate u'th order terms for the variance of phi^s.
    M is the number of external variables.

    :Definition: diploma thesis equation (3.97)

    :Yields: [ prefactor, multi-indices_r1_r2, inner_sum_indices,
               multi-index_r1, multi-index_r2, (u, p, q) ]

    "multi-indices_r1_r2" is needed for sum_parsing._eval_kronecker_deltas_k()

    :Example:
        >>> next(variance_phis(2, 1))
        [1/2, [1, 1], [(1, 2), (1, 2)], [1], [1], (2, 2, [1, 1])]
    """
    if DEBUG:
        print("u = %r, M = %r" % (u, M))
    # second sum p = 2, ..., 2u
    for p in range(2, 2*u + 1):
        if DEBUG:
            print("  p = %r" % p)
        # third sum |q| = p
        for q in multi_indices(p, 2, smallest=1):
            if DEBUG:
                print("    q = %r" % q)
            # fourth sum #r1 = q1
            for r1 in multi_indices_k(q[0], M):
                if DEBUG:
                    print("      r1 = %r" % r1)
                # fifth sum #r2 = q2
                for r2 in multi_indices_k(q[1], M):
                    if DEBUG:
                        print("        r2 = %r" % r2)
                    # sixth sum over index pairs
                    for factor, indices in inner_sum_indices(u, p):
                        # apply condition: \exists \phi such that i_\phi
                        # \in {1, ..., q1} and j_\phi \in {q1 + 1, ..., p}
                        condition = False
                        for i, j in indices:
                            if i <= q[0] and j > q[0]:
                                condition = True
                        if condition:
                            factor = Rational(factor, factorial(u))
                            final = [ factor, r1 + r2, indices, r1, r2, (u, p, q) ]
                            if DEBUG:
                                print("\t\t\t%r" % final)
                            yield final

#-----------------------------------------
def spectrum_R(u, M=1):
    """
    Generate u'th order terms for the matrix R(omega).
    M is the number of external variables.

    :Definition: diploma thesis equation (3.105)

    :Yields: [ prefactor, multi-indices_kn_l, inner_sum_indices,
               multi-indices_k, multi-index_l, (u, s, r, n, q) ]

    "multi-indices_kn_l" is needed for sum_parsing._eval_theta()

    :Example:
        >>> gen = spectrum_R(2, 1)
        >>> for i in range(2): next(gen) #doctest: +ELLIPSIS
        [1/2, ..., [(1, 2), (1, 2)], [[1, 1]], [], (2, 2, 0, 1, [2])]
        [1/2, ..., [(1, 2), (1, 2)], [[1], [1]], [], (2, 2, 0, 2, [1, 1])]

        # multi-indices_kn_l is ommited by the ELLIPSIS in the example
    """
    if DEBUG:
        print("u = %r, M = %r" % (u, M))
    # second sum s = 0, ..., 2u
    for s in range(2*u + 1):
        if DEBUG:
            print("  s = %r" % s)
        # third sum r = 0, ..., s
        for r in range(s + 1):
            if DEBUG:
                print("    r = %r" % r)
            # sum over n (number of A symbols)
            for n in range(s - r + 1):
                if DEBUG:
                    print("      n = %r" % n)
                # sum over |q| = s - r, #q = n
                for q in multi_indices(s - r, n, smallest=1):
                    if DEBUG:
                        print("        q = %r" % q)
                    # sum over #k1 = q1, ... #kn = qn, #l = r
                    for kn_l in multi_indices_kn_l(s, r, M):
                        # assign kn indices to k1, ..., kn
                        kn_indices = []
                        kn_copy = kn_l[:s-r]
                        l = kn_l[s-r:]
                        for q_i in q:
                            # indices for one multi-index k_i
                            k_i = []
                            while len(k_i) < q_i:
                                k_i.append(kn_copy.pop(0)[0])
                            kn_indices.append(k_i)
                        # inner sum over index pairs
                        for factor, indices in inner_sum_indices(u, s):
                            factor = Rational(factor, factorial(u))
                            final = [factor, kn_l, indices, kn_indices, l,
                                    (u, s, r, n, q)]
                            if DEBUG:
                                print("\t\t\t%s" % final)
                            yield final

#-----------------------------------------
def spectrum_ext(u, M=1):
    """
    Generate u'th order terms for the spectrum P_e(omega).
    M is the number of external variables.

    :Definition: diploma thesis equation (3.110)

    :Yields: [ prefactor, multi-indices_r1_r2, inner_sum_indices,
               multi-index_r1, multi-index_r2, (u, p, q) ]

    "multi-indices_r1_r2" is needed for sum_parsing._eval_theta()

    :Example:
        >>> next(spectrum_ext(2, 1))
        [1/2, [1, 1], [(1, 2), (1, 2)], [1], [1], (2, 2, [1, 1])]
    """
    return variance_phis(u, M)


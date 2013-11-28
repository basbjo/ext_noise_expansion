#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Helper functions using standard libraries.
"""

from math import factorial as _factorial

# memoization dictionaries
_MEMO_MULTI_INDICES = {}
_MEMO_MULTI_INDICES_KN_L = {}

#=========================================
# multi-indices and multinomial coefficients
def multi_indices(q, n, smallest=0):
    """
    Create a list of multi-indices k = (k_1, ..., k_n) such that
    k_1 + ... + k_n = q and k_i >= smallest for i = 1, ..., n.
    The minimum value smallest must be 0 or 1.

    :Returns: [ [k_1, ..., k_n], [k_1, ..., k_n], ... ]

    :Example:
        >>> multi_indices(3, 2)
        [[0, 3], [1, 2], [2, 1], [3, 0]]
        >>> multi_indices(4, 3, 1)
        [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    """
    if (q, n, smallest) in _MEMO_MULTI_INDICES:
        # if it was already calculated
        return _MEMO_MULTI_INDICES[(q, n, smallest)]
    # new calculation
    if n == 0 and q == 0:
        list_k = [[]]
    elif n == 0:
        list_k = []
    elif n == 0 and q == 0:
        list_k = [[]]
    elif n == 1 and q == 0 and smallest == 1:
        list_k = [[]]
    elif n == 1:
        list_k = [[q]]
    else:
        list_k = [ [k1] + prev_k
            for k1 in range(smallest, q-smallest*(n-1)+1)
                for prev_k in multi_indices(q-k1, n-1, smallest) ]
    # remember
    _MEMO_MULTI_INDICES[(q, n, smallest)] = list_k
    return  list_k

#-----------------------------------------
def multinomial_coefficient(tup):
    """
    Multinomial coefficient for a tuple of objects.
    No sorting (and no checks!).

    :Parameters:
        - `tup`: tuple

    :Returns: integer

    :Example:
        >>> multinomial_coefficient((1,2))
        2
        >>> multinomial_coefficient((1,1,2))
        3
        >>> multinomial_coefficient((1,2,1)) # no sorting
        6
        >>> multinomial_coefficient((1,1,1))
        1
    """
    if tuple(tup) == ():
        return 1
    numer = _factorial(len(tup))
    denom = 1
    count = 1
    tup = list(tup)
    prev = tup.pop(0)
    for i in tup:
        if i == prev:
            count += 1
        else:
            denom *= _factorial(count)
            prev = i
            count = 1
    denom *= _factorial(count)
    return int(numer/denom)

#-----------------------------------------
def multi_indices_kn_l(s, r, M):
    """
    Create all possible index tuples (k_1, ..., k_{s-r}, l) for A and C.

    The upper index can be 1 or 2 and is always 1 for A.

    :Parameters:
        - `s`: order in eta (external stochastic variables)
        - `r`: order of derivatives of C with respect to eta
        - `M`: number of components of eta

    :Returns: List of index lists

    :Example:
        >>> print(multi_indices_kn_l(2, 1, 2)) #doctest: +ELLIPSIS
        [[(1, 1), (1, 1)], [(1, 1), (1, 2)], ..., [(2, 1), (2, 2)]]
    """
    if (s, r, M) in _MEMO_MULTI_INDICES_KN_L:
        return _MEMO_MULTI_INDICES_KN_L[(s, r, M)]
    q = s - r
    if q == 0:
        if r == 0:
            list_i = [[]]
        else:
            list_i = [ [(i+1, j+1)] + prev_i
                        for i in range(M)
                        for j in range(2)
                        for prev_i in multi_indices_kn_l(s-1, r-1, M) ]
    else:
        list_i = [ [(i+1, 1)] + prev_i
                    for i in range(M)
                    for prev_i in multi_indices_kn_l(s-1, r, M) ]
    _MEMO_MULTI_INDICES_KN_L[(s, r, M)] = list_i
    return  list_i

#=========================================
# term merging

def merge_yield(l):
    """
    Merge consecutive list entries of a list or iterator if they are
    identical except for the first elements by summing the latter.

    :Returns: generator

    :Example:
        >>> l = [[3,1], [7,1], [5,2]]
        >>> [i for i in  merge_yield(l)]
        [[10, 1], [5, 2]]
    """
    last = []
    for i in l:
        if last and last[1:] == i[1:]:
            last[0] += i[0]
        else:
            if last:
                yield last
            last = i
    if last:
        yield last

#-----------------------------------------
def sort_merge(l):
    """
    Merge list entries of a list or iterator if they are identical
    except for the first elements by summing the latter.
    Consecutive list entries for which the last elements are identical
    are previously sorted.  The last elements are omitted.

    :Returns: generator

    :Example:
        >>> for i in sort_merge([
        ... [2, 1, 1],
        ... [1, 2, 1],
        ... [3, 1, 1],
        ... [2, 1, 2] ]): print(i)
        [5, 1]
        [1, 2]
        [2, 1]
    """
    def _sort_merge(l):
        """omit last element, sort and merge"""
        for item in l:
            item.pop(-1)
        l.sort(key=lambda i: i[1:])
        return merge_yield(l)
    part = []
    for i in l:
        # group list entries for sorting
        if part and part[-1][-1] == i[-1]:
            # append if the last element is still the same
            part.append(i)
        else:
            if part:
                for final in _sort_merge(part):
                    yield final
            part = [i]
    for final in _sort_merge(part):
        yield final


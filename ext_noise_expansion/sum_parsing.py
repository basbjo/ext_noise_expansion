#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Generators for sum parsing.
"""

from .tools_universal import merge_yield, sort_merge

#-----------------------------------------
def mean(gen):
    """
    Partially evaluate and simplify the sum for the mean.

    :Definition: diploma thesis equation (3.90)

    :Yields: [ prefactor, phis_indices, epsilon_squared_indices ]

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> gen = sum_generation.mean(2, 2)
        >>> for item in mean(gen): item
        [1/2, (1, 1), (1, 1)]
        [1/2, (2, 2), (2, 2)]
        [3, (1, 1, 1), (1, 1)]
        [3, (2, 2, 2), (2, 2)]
        [3, (1, 1, 1, 1), (1, 1)]
        [6, (1, 1, 2, 2), (1, 2)]
        [3, (2, 2, 2, 2), (2, 2)]
    """
    def unmerged(gen):
        """sort commutative index tuples"""
        for factor, multi_k, tuples, u_q in _eval_kronecker_deltas_k(gen):
            multi_k = multi_k[:]
            tuples = tuples[:]
            multi_k.sort() # commutative derivatives
            tuples.sort() # commutative product of epsilon_i^2
            yield [factor, tuple(multi_k), tuple(tuples), u_q]
    return sort_merge(merge_yield(unmerged(gen)))

#-----------------------------------------
def variance_xi(gen):
    """
    Partially evaluate and simplify the sum for the variance of xi.

    :Definition: diploma thesis equation (3.91)

    :Yields: [ prefactor, C_indices, epsilon_squared_indices ]

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> gen = variance_xi(sum_generation.variance_xi(2, 2))
        >>> for i in range(6): next(gen)
        [1/2, ((1, 1), (1, 1)), (1, 1)]
        [1, ((1, 1), (1, 2)), (1, 1)]
        [1/2, ((1, 2), (1, 2)), (1, 1)]
        [1/2, ((2, 1), (2, 1)), (2, 2)]
        [1, ((2, 1), (2, 2)), (2, 2)]
        [1/2, ((2, 2), (2, 2)), (2, 2)]
    """
    def unmerged(gen):
        """sort commutative index tuples"""
        for factor, multi_l, tuples, u_r in _eval_kronecker_deltas_l(gen):
            multi_l = multi_l[:]
            tuples = tuples[:]
            multi_l.sort() # commutative derivatives
            tuples.sort() # commutative product of epsilon_i^2
            yield [factor, tuple(multi_l), tuple(tuples), u_r]
    return sort_merge(merge_yield(unmerged(gen)))

#-----------------------------------------
def variance_phis(gen):
    """
    Partially evaluate and simplify the sum for the variance of phis.

    :Definition: diploma thesis equation (3.97)

    :Yields: [ prefactor, r1_indices, r2_indices, epsilon_squared_indices ]

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> gen = variance_phis(sum_generation.variance_phis(2, 2))
        >>> for i in range(6): next(gen)
        [1/2, (1,), (1,), (1, 1)]
        [1/2, (2,), (2,), (2, 2)]
        [3, (1,), (1, 1), (1, 1)]
        [3, (2,), (2, 2), (2, 2)]
        [3, (1, 1), (1,), (1, 1)]
        [3, (2, 2), (2,), (2, 2)]
    """
    def unmerged(gen):
        """sort commutative index tuples"""
        for factor, _, tuples, multi_r1, multi_r2, u_p_q \
                in _eval_kronecker_deltas_k(gen):
            # multi-indices_r1_r2 was omitted as _
            multi_r1 = multi_r1[:]
            multi_r2 = multi_r2[:]
            tuples = tuples[:]
            multi_r1.sort() # commutative derivatives
            multi_r2.sort() # commutative derivatives
            tuples.sort() # commutative product of epsilon_i^2
            yield [factor, tuple(multi_r1), tuple(multi_r2),
                    tuple(tuples), u_p_q]
    return sort_merge(merge_yield(unmerged(gen)))

#-----------------------------------------
def spectrum_R(gen):
    """
    Partially evaluate and simplify the sum for the matrix R(omega).

    :Definition: diploma thesis equation (3.105)

    :Yields: [ prefactor, theta_indices, [A_indices, ...], C_indices ]

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> gen = spectrum_R(sum_generation.spectrum_R(2, 2))
        >>> for i in range(6): next(gen)
        [1/2, ((((1, 1), (1, 1)), ((1, 1), (1, 1))), 1), [(1, 1)], ()]
        [1/2, ((((2, 1), (2, 1)), ((2, 1), (2, 1))), 1), [(2, 2)], ()]
        [1/2, ((((1, 1), (1, 1)), ((1, 1), (1, 1))), 2), [(1,), (1,)], ()]
        [1/2, ((((2, 1), (2, 1)), ((2, 1), (2, 1))), 2), [(2,), (2,)], ()]
        [1/2, ((((1, 1), (1, 1)), ((1, 1), (1, 1))), 1), [(1,)], ((1, 1),)]
        [1/2, ((((1, 1), (1, 2)), ((1, 1), (1, 2))), 1), [(1,)], ((1, 2),)]
    """
    def unmerged(gen):
        """sort commutative index tuples"""
        for factor, _, tuples, multi_kn, multi_l, usrnq \
                in _eval_theta(gen):
            # multi-indices_kn_l was omitted as _
            tuples = tuples[:]
            multi_kn = multi_kn[:]
            multi_l = multi_l[:]
            for i, k_i in enumerate(multi_kn):
                k_i.sort() # commutative derivatives
                multi_kn[i] = tuple(k_i)
            multi_l.sort() # commutative derivatives
            tuples.sort() # commutative product of theta
            tuples = [tuple(tuples), usrnq[3]] # n = usrnq[3]
            yield [factor, tuple(tuples), multi_kn, tuple(multi_l), usrnq]
    return sort_merge(merge_yield(unmerged(gen)))

#-----------------------------------------
def spectrum_ext(gen):
    """
    Partially evaluate and simplify the sum for the spectrum P_e(omega).

    :Definition: diploma thesis equation (3.110)

    :Yields: [ prefactor, r1_indices, r2_indices, Theta_indices, u_p_q ]

    :Example:
        >>> from ext_noise_expansion import sum_generation
        >>> gen = spectrum_ext(sum_generation.spectrum_ext(2, 2))
        >>> for i in range(6): next(gen)
        [1/2, (1,), (1,), (((1, 1), (1, 2)), ((1, 1), (1, 2)))]
        [1/2, (2,), (2,), (((2, 1), (2, 2)), ((2, 1), (2, 2)))]
        [2, (1,), (1, 1), (((1, 1), (1, 1)), ((1, 1), (1, 2)))]
        [1, (1,), (1, 1), (((1, 1), (1, 2)), ((1, 1), (1, 2)))]
        [2, (2,), (2, 2), (((2, 1), (2, 1)), ((2, 1), (2, 2)))]
        [1, (2,), (2, 2), (((2, 1), (2, 2)), ((2, 1), (2, 2)))]
    """
    def unmerged(gen):
        """sort commutative index tuples"""
        for factor, _, tuples, multi_r1, multi_r2, u_p_q \
                in _eval_theta(gen, lowercase=False):
            # multi-indices_r1_r2 was omitted as _
            multi_r1 = multi_r1[:]
            multi_r2 = multi_r2[:]
            tuples = tuples[:]
            multi_r1.sort() # commutative derivatives
            multi_r2.sort() # commutative derivatives
            tuples.sort() # commutative product of epsilon_i^2
            yield [factor, tuple(multi_r1), tuple(multi_r2),
                    tuple(tuples), u_p_q]
    return sort_merge(merge_yield(unmerged(gen)))

#=========================================
# index creation

#-----------------------------------------
def _eval_kronecker_deltas_k(gen):
    """
    Partially evaluate \\Delta^{k}_{ij} symbols.

    :Definition: diploma thesis equations (3.43) and (3.89)

    :Yields: [ prefactor, multi-index_k, Delta_indices, ... ]

    :Example:
        >>> gen = _eval_kronecker_deltas_k([
        ... [1, [2, 1, 2], [(1, 2), (1, 3)]],
        ... [1, [2, 1, 2], [(1, 3), (1, 3)]] ])
        >>> for item in gen: item
        [1, [2, 1, 2], [2, 2]]
    """
    for item in gen:
        factor = item.pop(0)
        multi_k = item.pop(0)
        tuples = item.pop(0)
        new_tuples = []
        iszero = False
        for tup in tuples:
            # substitute tuples by multi-index entries
            tup = [multi_k[tup[0]-1], multi_k[tup[1]-1]]
            if len(set(tup)) > 1:
                # many Kronecker deltas evaluate to zero
                iszero = True
            new_tuples.append(tup[0])
        # yield non-zero terms
        if not iszero:
            new_item = [factor, multi_k, new_tuples] + item
            yield new_item

#-----------------------------------------
def _eval_kronecker_deltas_l(gen):
    """
    Partially evaluate \\Delta^{l}_{ij} symbols.

    :Definition: diploma thesis equations (3.44) and (3.91)

    :Yields: [ prefactor, multi-index_l, Delta_indices, ... ]

    :Example:
        >>> gen = _eval_kronecker_deltas_l([
        ... [1, [(2, 1), (1, 1), (2, 2)], [(1, 2), (1, 3)]],
        ... [1, [(2, 1), (1, 1), (2, 2)], [(1, 3), (1, 3)]] ])
        >>> for item in gen: item
        [1, [(2, 1), (1, 1), (2, 2)], [2, 2]]
    """
    for item in gen:
        factor = item.pop(0)
        multi_l = item.pop(0)
        tuples = item.pop(0)
        new_tuples = []
        iszero = False
        for tup in tuples:
            # substitute tuples by multi-index entries
            tup = [multi_l[tup[0]-1][0], multi_l[tup[1]-1][0]]
            if len(set(tup)) > 1:
                # many Kronecker deltas evaluate to zero
                iszero = True
            new_tuples.append(tup[0])
        # yield non-zero terms
        if not iszero:
            new_item = [factor, multi_l, new_tuples] + item
            yield new_item

#-----------------------------------------
def _eval_theta(gen, lowercase=True):
    """
    Partially evaluate theta (Theta) symbols for spectrum_R() if lowercase
    and for spectrum_ext() if not lowercase.

    :Definition:
        diploma thesis equations (3.104) and (3.109)

    :Yields: [ prefactor, multi-indices_[kn_l|r1_r2], theta_indices, ... ]

    :Example:
        >>> gen = _eval_theta([
        ... [1, [(2, 1), (1, 1), (2, 2)], [(1, 2), (1, 3)]],
        ... [1, [(2, 1), (1, 1), (2, 2)], [(1, 3), (1, 3)]] ])
        >>> for item in gen: item
        [1, [(2, 1), (1, 1), (2, 2)], [((2, 1), (2, 2)), ((2, 1), (2, 2))]]

        # if not lowercase, the last list elements must be u_p_q, q = [q1, q2]
        >>> gen = _eval_theta([
        ... [1, [2, 2, 1], [(1, 2), (1, 3)], (2, 3, [1, 2])],
        ... [1, [2, 2, 1], [(1, 2), (1, 2)], (2, 3, [2, 1])],
        ... [1, [2, 2, 1], [(1, 2), (1, 2)], (2, 3, [1, 2])] ], False)
        >>> for item in gen: item
        [1, [2, 2, 1], [((2, 1), (2, 1)), ((2, 1), (2, 1))], (2, 3, [2, 1])]
        [1, [2, 2, 1], [((2, 1), (2, 2)), ((2, 1), (2, 2))], (2, 3, [1, 2])]
    """
    for item in gen:
        factor = item.pop(0)
        multi_indices = item.pop(0)
        tuples = item.pop(0)
        new_tuples = []
        iszero = False
        if not lowercase:
            q1, q2 = item[-1][2]
            # upper indices, see diploma thesis equation (3.98)
            upper_indices = q1*[1] + q2*[2]
        for tup in tuples:
            # substitute tuples by multi-index entries
            if lowercase:
                test = [multi_indices[tup[0]-1][0], multi_indices[tup[1]-1][0]]
            else:
                test = [multi_indices[tup[0]-1], multi_indices[tup[1]-1]]
            if len(set(test)) > 1:
                # many Kronecker deltas evaluate to zero
                iszero = True
            if lowercase:
                tup1 = list(multi_indices[tup[0]-1])
                tup2 = list(multi_indices[tup[1]-1])
            else:
                tup1 = [ multi_indices[tup[0]-1], upper_indices[tup[0]-1] ]
                tup2 = [ multi_indices[tup[1]-1], upper_indices[tup[1]-1] ]
            if tup1[1] == tup2[1]:
                # the specific value is unimportant
                tup1[1] = 1
                tup2[1] = 1
            tup = [tuple(tup1), tuple(tup2)]
            tup.sort() # the order is unimportant
            tup = tuple(tup)
            new_tuples.append(tup)
        # yield non-zero terms
        if not iszero:
            new_item = [factor, multi_indices, new_tuples] + item
            yield new_item


#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Solvers for the stationary state.
"""

from sympy import Matrix, solve
from .tools_sympy import def_nprint

#-----------------------------------------
def simple_solve(g, phi, select=None, pretty=False):
    """
    Solve g(phi) = 0 by solving g_i(phi_i) = 0 iteratively.
    Negative solutions are dropped if possible.

    :Parameters:
        - `g`: Matrix of reaction rate equations as function of phi
        - `phi`: Matrix of components of phi, e.g. Matrix([phi1, phi2])
        - `select`: integer to select the solution if ambiguous; if select
                    is not an integer, it will be obtained from user input
        - `pretty`: use pprint instead of print when asking for user input

    :Raises: `IndexError` if select is out of range.

    :Returns: Matrix of stationary concentrations phi or None.
    """
    g = Matrix(g)
    phi = Matrix(phi)
    subs_dicts = []
    # first variable
    for phi0s in solve(g[0], phi[0]):
        subs_dicts.append({phi[0]: phi0s})

    # further variables
    for i in range(1, len(phi)):
        new_dicts = []
        for subs_dict in subs_dicts:
            # for all solutions for phi[i]
            for phiis in solve(g.subs(subs_dict)[i], phi[i]):
                new_dict = subs_dict.copy()
                for key, item in new_dict.items():
                    # substitute in phis[<i]
                    new_dict[key] = item.subs({phi[i]: phiis})
                # append phis[i] to dict
                new_dict.update({phi[i]: phiis})
                new_dicts.append(new_dict)
        # extended dictionary
        subs_dicts = new_dicts
    solutions = []
    for subs_dict in subs_dicts:
        solutions.append(phi.subs(subs_dict))
    # remove negative solutions
    solutions = list(filter(lambda i: # each matrix
            not any(i.applyfunc(lambda j: # each matrix entry
                j.is_negative)), solutions))
    # return or select solution
    if len(solutions) == 0:
        return None
    elif len(solutions) == 1:
        return solutions[0]
    elif isinstance(select, int):
        try:
            solution = solutions[select]
        except IndexError:
            raise IndexError("There are only %d solutions (select = %d)."
                    % (len(solutions), select))
        return solution
    else:
        # print solutions
        print("There is more than one solution:\n")
        nprint = def_nprint(pretty)
        for i, solution in enumerate(solutions):
            print("%3d:" % i)
            nprint(solution.evalf())
        user_input = None
        # ask user to select a solution
        while not user_input:
            user_input = True
            select = input("Please select a solution (integer < %d): " %
                    len(solutions))
            try:
                select = int(select)
            except ValueError:
                user_input = False
            if not select in range(len(solutions)):
                user_input = False
        # return selected solution
        return solutions[select]


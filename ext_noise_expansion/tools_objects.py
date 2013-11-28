#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Dummy matrix object for ReactionSystemBase and DefinitionError.
"""

from sympy import Symbol, Matrix

#-----------------------------------------
class DefinitionError(Exception):
    """
    System definition error.
    """
    def __init__(self, string=None):
        """
        System definition error.
        """
        Exception.__init__(self)
        if string:
            self.string = string
        else:
            self.string = None
    def __str__(self):
        if self.string:
            return self.string
        else:
            return "The reaction system is not fully defined."

#-----------------------------------------
class EigenvalueError(DefinitionError):
    """
    System definition error.
    """
    def __init__(self, string=None):
        """
        System definition error.
        """
        DefinitionError.__init__(self)
        if string:
            self.string = string
        else:
            self.string = None
    def __str__(self):
        if self.string:
            return self.string
        else:
            return "The reaction system is not fully defined."

#-----------------------------------------
class DummySymbol(Matrix):
    """
    Dummy matrix object compatible with tools_sympy.taylor_coeff().
    """
    def __init__(self, label, indices, eta_dict):
        """
        Dummy matrix object compatible with tools_sympy.taylor_coeff().

        :Example:
            >>> eta1 = Symbol('eta1', positive=True)
            >>> A = DummySymbol('A', [0], {1: eta1})
            >>> A[0]
            A[0]
            >>> A.diff(eta1)[0]
            A[1]
            >>> A.diff(eta1, eta1)[0]
            A[1, 1]
        """
        indices = tuple(indices)
        if label == 'C' and indices == (0,):
            # C depends on two variables
            indices = (0, 0)
        matrix = [Symbol(label+str(list(indices)), positive=True)]
        self.eta_dict = eta_dict
        Matrix.__init__(self, matrix)
        self.label = label
        if indices == (0,) or indices == (0, 0):
            self.ind = ()
        else:
            self.ind = indices

    def __new__(cls, label, indices, eta_dict):
        indices = tuple(indices)
        del(eta_dict)
        if label == 'C' and indices == (0,):
            # C depends on two variables
            indices = (0, 0)
        matrix = [Symbol(label+str(list(indices)), positive=True)]
        new_instance = Matrix.__new__(cls, matrix)
        return new_instance

    def diff(self, *args):
        """
        Emulate derivatives by appending indices.
        """
        eta_dict = self.eta_dict
        indices = self.ind
        for key in args:
            eta_key = list(eta_dict.values()).index(key)
            indices += ((list(eta_dict.keys())[eta_key]),)
        return DummySymbol(self.label, indices, self.eta_dict)

    def subs(self, *args):
        """
        Does nothing to leave the symbolic representation unchanged.
        """
        return self


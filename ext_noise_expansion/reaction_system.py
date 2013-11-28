#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Reaction system definition and derivatives for the series expansion.

Examples
========

Symbolic test system:
    >>> rs = ReactionSystemBase(3)
    >>> rs.eval_symbol('A', ())[0]
    A[0]
    >>> rs.eval_symbol('B', (2, 2))[0]
    B[2, 2]
    >>> rs.eval_symbol('C', ((1, 2),))[0]
    C[(1, 2)]

    # indices are sorted
    >>> rs.eval_symbol('phis', (2, 1))[0]
    phis[1, 2]

    # variances and correlation times
    >>> list(rs.etavars)
    [epsilon1**2, epsilon2**2, epsilon3**2]
    >>> list(rs.etaKs)
    [K1, K2, K3]

    # products of variances
    >>> rs.eval_variances((1, 2))
    epsilon1**2*epsilon2**2

    # theta symbol for the calculation of R
    >>> list(rs.theta_R(((((1, 1), (1, 1)), ((1, 1), (1, 2))), 1)))
    [epsilon1**4/(-A[0] + K1 + I*omega)**2]

    # Theta symbol for the calculation of P_e
    >>> rs.theta_P((((1, 1), (1, 1)), ((1, 1), (1, 2))))
    K1*epsilon1**4/(pi*(K1**2 + omega**2))

Load definition from file and use a solver for stationary state:
    >>> from ext_noise_expansion import simple_solve
    >>> rs = ReactionSystem.from_string(yaml_file='test_system.yaml')
    >>> rs.eval_at_phis(solver=simple_solve)

Load definition from file and externally solve for phis:
    >>> from sympy import solve
    >>> rs = ReactionSystem.from_string(yaml_file='test_system.yaml')
    >>> phis = solve(rs.g[0], rs.phi[0])
    >>> rs.eval_at_phis(phis)

    see also definition_parser.py and ext_noise_expansion.rst

Numerical evaluation:
    >>> from sympy import factor # for python/python3 compatibility
    >>> new = rs.copy() # make a copy
    >>> new.num_eval({'k': 5, 'l':3}, ifevalf=False)
    >>> [factor(new.A[0])]
    [-5*(eta + 1)]
    >>> [factor(rs.A[0])] # unchanged
    [-k*(eta + 1)]
"""

from sympy import Symbol, Matrix, eye, zeros, factorial, I, pi, sympify
from sympy.core.sympify import SympifyError
from .tools_sympy import diag, matsimp, taylor_coeff, def_nprint
from .tools_sympy import lyapunov_equation, lyapunov_equation_2, def_N
from .tools_objects import DefinitionError, EigenvalueError, DummySymbol
from .tools_universal import multinomial_coefficient
from .definition_parser import string_parser

K = "K" # symbol base for inverse correlation times
ETA = "eta" # symbol base for extrinsic variables
EPS = "epsilon" # symbol base for standard deviations
SIZE = "Omega"  # symbol for the system size
FREQ = "omega"  # symbol for the angular frequency
SYMBOLS = ['phis', 'A', 'B', 'C'] # evaluated symbols
DATA_KEYS = ['phi', 'phis', 'eta', 'etavars', 'etaKs', 'S', 'f', 'g', 'A',
        'C', 'B', 'DM', 'omega', 'Omega', 'map_dict']

#-----------------------------------------
def _lyapunov_equation_C(self, A, B):
    """
    Tries to calculate C by means of a slightly generalized Lyapunov equation.
    This is not efficient, since diagonalization is in general impossible.

    :Returns: matrix 'C' or 'None'
    """
    A1 = A
    A2 = A
    B1 = B
    B2 = B
    for j, etaj in enumerate(self.eta):
        A1 = A1.subs({etaj: self.eta1[j]})
        A2 = A2.subs({etaj: self.eta2[j]})
        B1 = B1.subs({etaj: self.eta1[j]})
        B2 = B2.subs({etaj: self.eta2[j]})
    try:
        C = lyapunov_equation_2(A1, A2, B1*B2.T)
        return C
    except ValueError:
        return None

#-----------------------------------------
class ReactionSystemBase(object):
    """
    Basic class to define a reaction system and calculate Taylor
    coefficients.
    """
    def __init__(self, M, omega=FREQ, Omega=SIZE):
        """
        Basic class to define a reaction system and calculate Taylor
        coefficients.  It can be used to create symbolical output.

        :Parameters:
            - `M`: number of extrinsic stochastic variables
            - `pi`: symbol for pi
            - `omega`: frequency symbol string for the spectrum
            - `Omega`: symbol string for the system size parameter

        :Raises: `ValueError` if M is not a positive integer

        :Example: see module docstring
        """
        if not (isinstance(M, int) and M > 0):
            raise ValueError("M must be a positive integer")
        # M extrinsic variables with corresponding variances
        self.M = M
        self.pi = pi
        self.omega = Symbol(omega, positive=True) # frequency symbol
        self.Omega = Symbol(Omega, positive=True) # symbol for the system size
        self.eta = [] # stochastic variables
        self.etavars = [] # variances
        self.etaKs = [] # inverse correlations times
        self.TH_kernel = None # Matrix kernel of theta_R
        self.TH_K = Symbol("_XxXx") # symbol for sum of inv corr times
        for i in range(self.M):
            self.eta.append(Symbol(ETA+str(i+1), positive=True))
            self.etavars.append(Symbol(EPS+str(i+1), positive=True)**2)
            self.etaKs.append(Symbol(K+str(i+1), positive=True))
        self.eta = Matrix([self.eta])
        self.etavars = Matrix([self.etavars])
        self.etaKs = Matrix([self.etaKs])
        # dictionaries for eta with and without upper indices
        self.eta1 = []
        self.eta2 = []
        self.eta_dict = {}
        self.etai_dict = {}
        self._update_eta_dicts()
        # dummy symbols
        join_dicts = {}
        join_dicts.update(self.eta_dict)
        join_dicts.update(self.etai_dict)
        for label in SYMBOLS:
            setattr(self, label, DummySymbol(label, [0], join_dicts))
            setattr(self, "memo_"+label, {})
        # be sure symbols needed by other modules do exist
        self.phis = DummySymbol('phis', [0], join_dicts)
        self.A = DummySymbol('A', [0], join_dicts)
        self.C = DummySymbol('C', [0], join_dicts)
        self.memo_C = {}

    #-----------------------------------------
    def _update_eta_dicts(self):
        """(re)create dictionaries of eta symbols"""
        self.eta_dict = dict(enumerate(self.eta, 1))
        # eta with upper indices for different times
        self.eta1 = []
        self.eta2 = []
        self.etai_dict = {}
        for i in self.eta:
            self.eta1.append(Symbol(str(i)+"__1", positive=True))
            self.eta2.append(Symbol(str(i)+"__2", positive=True))
        for i, _ in enumerate(self.eta, 1):
            self.etai_dict[(i, 1)] = self.eta1[i-1]
            self.etai_dict[(i, 2)] = self.eta2[i-1]

    #-----------------------------------------
    def copy(self):
        """
        Returns a copy of self.
        """
        new = ReactionSystemBase(self.M)
        for key, item in self.__dict__.items():
            setattr(new, key, item)
        return new

    #-----------------------------------------
    def eval_symbol(self, symbol, tup, chop_imag=None):
        """
        Returns symbols or values for Taylor coefficients.

        :Parameters:
            - `symbol`: symbol as 'phis', 'A', 'C' (see SYMBOLS)

            - `tup`: index tuple such as () or (n,) where n denotes a
                  derivative with respect to the n'th extrinsic field
                  (starting with 1, the n'th field is self.eta[n-1]),

                  exception: indices as ((1, 1), (1, 2), (2, 1)) for C

                  special case: (0,) is treated identically to ()

        :Raises: `KeyError` for wrong keys.

        :Example: see module docstring
        """
        if tup == (0,):
            tup = ()
        tup = tuple(sorted(tup))
        if symbol == "C" and not self.C:
            # determine Taylor coefficients iteratively
            return self._Taylor_C(tup, chop_imag=chop_imag)*factorial(len(tup))
        elif symbol == "C":
            # eta with upper indices
            eta_dict = self.etai_dict
        else:
            # eta without upper indices
            eta_dict = self.eta_dict
        func = getattr(self, symbol)
        memo = getattr(self, "memo_"+symbol)
        try:
            c = taylor_coeff(func, eta_dict, tup, memo=memo)
        except KeyError as args:
            raise KeyError("Unknown key for Taylor coefficient of "
                    +"%s: %s" % (symbol, args))
        return c

    #-----------------------------------------
    def theta_R(self, tuples):
        """
        theta symbols for the calculation of R(omega).

        :Parameters:
            - `tuples` tuples as ((((1, 1), (1, 1)), ((1, 1), (1, 2))), 1)

        :Example: see module docstring

        :Raises: `IndexError` for lower indices smaller 1.
        """
        tuples = list(tuples)
        n = tuples.pop(-1)
        sumK = 0 # sum of inverse corelation times
        factor = 1 # product of variances
        tuples = list(tuples[0])
        if tuples == [0]: # allows for (0,) notation
            tuples = []
        for tup in tuples:
            # Kronecker delta
            if tup[0][0] != tup[1][0]:
                return zeros(*self.A.rows)
            # variances
            i = tup[0][0]
            if i > 0:
                factor *= self.etavars[tup[0][0]-1]
            else:
                raise IndexError("Lower indices must be"
                        +"larger or equal than 1.")
            # invers correlation times
            if tup[0][1] != tup[1][1]:
                sumK += self.etaKs[tup[0][0]-1]
        # matrix kernel
        if not self.TH_kernel:
            A0 = self.eval_symbol('A', ())
            self.TH_kernel = ( - A0 +
                    eye(self.A.rows)*(self.TH_K + I*self.omega)).inv()
        # return
        return self.TH_kernel.subs({self.TH_K: sumK})**(1 + n) * factor

    #-----------------------------------------
    def theta_P(self, tuples):
        """
        Theta symbols for the calculation of P_e(omega).

        :Parameters:
            - `tuples` tuples as (((1, 1), (1, 1)), ((1, 1), (1, 2)))

        :Example: see module docstring

        :Raises: `IndexError` for lower indices smaller 1.
        """
        tuples = list(tuples)
        sumK = 0 # sum of inverse corelation times
        factor = 1 # product of variances
        for tup in tuples:
            # Kronecker delta
            if tup[0][0] != tup[1][0]:
                return zeros(*self.A.rows)
            # variances
            i = tup[0][0]
            if i > 0:
                factor *= self.etavars[tup[0][0]-1]
            else:
                raise IndexError("Lower indices must be"
                        +"larger or equal than 1.")
            # invers correlation times
            if tup[0][1] != tup[1][1]:
                sumK += self.etaKs[tup[0][0]-1]
        # return
        return factor*sumK/pi/(sumK**2 + self.omega**2)

    #-----------------------------------------
    def eval_variances(self, tup):
        """
        Returns symbols or values for products of epsilon_i^2.

        :Parameters:
            - `tup`: index tuple such as () or (1,2) where the integers
                  denote the variances in self.etavars (the variance that
                  corresponds to the n'th variable is self.etavars[n-1])

        :Example: see module docstring
        """
        r = 1
        for i in tup:
            r *= self.etavars[i-1]
        return r

    #-----------------------------------------
    def _Taylor_C(self, tup, chop_imag=None):
        """
        Taylor coefficients of C divided by the factorial of the order.

        :Definition: diploma thesis equations (3.30), (3.117)

        :Parameters:
            - `tup`: index tuple as ((1, 1), (1, 2), (2, 1)),
                  special case: (0,) is treated identically to ()
            - `chop_imag`: Omit imaginary parts if abs(imag/real) < chop_imag
                  in the calculation of the Taylor coefficients of C. If
                  chop_imag is True, a default value is used.

        :Raises: `KeyError` for wrong keys.

        :Example: see module docstring (called by self.eval_symbol())
        """
        if tup in self.memo_C:
            # result from memoization dictionary
            return self.memo_C[tup]

        else:
            # iteratively determine coefficients by
            # expansion of the Lyapunov equation
            coeff = multinomial_coefficient(tup)
            A0 = self.eval_symbol('A', ())
            D = None # second matrix in Lyapunov equation
            ind_1 = ()
            ind_2 = ()
            for i in tup:
                if   i[1] == 1:
                    ind_1 += (i,)
                elif i[1] == 2:
                    ind_2 += (i,)
                else:
                    raise KeyError("Unknown upper index for eta: %s." % i[1])
            # B * B.T term
            ind_B1 = [i[0] for i in ind_1]
            ind_B2 = [i[0] for i in ind_2]
            B1 = self.eval_symbol('B', ind_B1) * multinomial_coefficient(ind_1)
            B2 = self.eval_symbol('B', ind_B2) * multinomial_coefficient(ind_2)
            D = B1 * B2.T
            # A * C terms
            ind_A = list(ind_1)
            ind_C = list(ind_2)
            for i in range(len(ind_A)):
                if i > 0:
                    ind_C.insert(0, ind_A.pop(-1))
                D += (   self.eval_symbol('A', [i[0] for i in ind_A])
                       * self._Taylor_C(tuple(ind_C), chop_imag=chop_imag)
                       * multinomial_coefficient(ind_A)
                       * multinomial_coefficient(ind_C))
            # C * A.T terms
            ind_A = list(ind_2)
            ind_C = list(ind_1)
            for i in range(len(ind_A)):
                if i > 0:
                    ind_C.append(ind_A.pop(0))
                D += (   self._Taylor_C(tuple(ind_C), chop_imag=chop_imag)
                       * self.eval_symbol('A', [i[0] for i in ind_A]).T
                       * multinomial_coefficient(ind_A)
                       * multinomial_coefficient(ind_C))
            # Lyapunov equation
            func = lyapunov_equation(coeff*A0, D, chop_imag=chop_imag)
            self.memo_C.update({tup: func})
            return func

#-----------------------------------------
class ReactionSystem(ReactionSystemBase):
    """
    Class to define a reaction system and calculate Taylor coefficients.
    """
    # symbols to be evaluated by self.eval_at_phis:
    _EVAL = ["eta", "S", "f", "g", "A", "C", "B", "DM"]

    #-----------------------------------------
    def __init__(self, data, map_dict=False, C_attempt=False,
            omega=FREQ, Omega=SIZE, verbose=0):
        """
        Class to define a reaction system and calculate Taylor coefficients.
        The matrices 'A' and, if possible, 'C' as well as intermediate results
        are calculated if they are not provided directly.

        :Parameters:
            - `data`: Dictionary that defines the reaction system.
            - `map_dict`: Optional dictionary to map strings to symbols.
                    May as well be added to the 'data' dictionary.
            - `C_attempt`: If True, the calculation of C is attempted.
                    This is in general not possible and may be
                    unnecessarily time consuming.
            - `omega`: frequency symbol string for the spectrum
            - `Omega`: symbol string for the system size parameter
            - `verbose`: print phis, A, B, DM and C (0: not at all, or
                    with 1: print, 2: sympy pprint, 3: IPython display)

        The following keys are accepted in 'data' (see DATA_KEYS).
            - `phi`:  Vector of symbols for macroscopic concentrations
            - `phis`: Macroscopic stationary state concentrations
            - `eta`:  Vector of symbols for external fluctuations
            - `etavars`: Vector of variances of the external fluctuations
            - `etaKs`: Vector of inverse correlation times of the external
                       fluctuations
            - `S`:    Stoichiometric matrix
            - `f`:    Macroscopic transition rates f_i({phi_j}, {eta_k})
            - `g`:    Reaction rate equations g_i({phi_j}, {eta_k})
            - `A`:    Jacobian matrix
            - `C`:    Lyapunov matrix
            - `B`:    Cholesky matrix
            - `DM`:   Diffusion matrix

        One of the following sets should be provided to unambiguously
        define a reaction system at macroscopic stationary state.
        
          - eta, phi, f(phi), S
          - eta, phi, g(phi), C or B or DM
          - eta, A, C or B or DM, optionally phi
          - eta, A, S, f, optionally phi

        If any function depends on phi, the latter should be provided to
        have the function evaluated at stationary concentration phis.
        Therefor, self.eval_at_phis() should be called.

        :Raises: `DefinitionError` if system definition fails.

        :Example: see module docstring
        """
        eta = data.get('eta')
        if not eta:
            raise DefinitionError("'eta' is not defined")
        for key in data.keys():
            if not key in DATA_KEYS:
                raise DefinitionError("Key '%s' is not recognized." % key)
        ReactionSystemBase.__init__(self, len(eta), omega=omega, Omega=Omega)
        self.phis = None
        self.eta = eta
        self.C_attempt = C_attempt

        # read data
        phi     = data.get('phi')
        phis    = data.get('phis')
        S       = data.get('S')
        f       = data.get('f')
        g       = data.get('g')
        A       = data.get('A')
        C       = data.get('C')
        B       = data.get('B')
        DM      = data.get('DM')
        # remove keys without value
        remove_keys = []
        for key, value in data.items():
            if not value:
                remove_keys.append(key)
        for key in remove_keys:
            data.pop(key)
        # get optional keys
        self.etavars = data.get('etavars', self.etavars)
        self.etaKs   = data.get('etaKs', self.etaKs)
        self.omega   = data.get('omega', self.omega)
        self.Omega   = data.get('Omega', self.Omega)
        map_dict = data.get('map_dict', {})
        self.map_dict = {}
        # add keys that might be missing in map_dict
        for i in list(self.etavars) \
                + list(self.etaKs) \
                + [self.omega, self.Omega]:
            symbol = i.as_base_exp()[0]
            self.map_dict.update({symbol.name: symbol})
        # map_dict has priority
        self.map_dict.update(map_dict)

        # dictionaries for eta with and without upper indices
        ReactionSystemBase._update_eta_dicts(self)

        # Jacobian at stationary state
        if not A:
            if not phi:
                error = "'A' cannot be determined "
                if S and f or g:
                    error += "('phi' is missing)"
                else:
                    error = "'A' is missing"
                raise DefinitionError(error)
            else:
                if not g:
                    if S and f:
                        g = S*f
                        A = g.jacobian(phi)
                    else:
                        error = "'A' cannot be determined "
                        if S or f:
                            error += "('S' or 'f' is missing)"
                        elif B or C or DM:
                            error += "('g' is missing)"
                        else:
                            error += "('A' or 'S' and 'f' are missing)"
                        raise DefinitionError(error)
                else:
                    A = g.jacobian(phi)

        # Lyapunov matrix at stationary state
        if not C and not B:
            if not DM:
                if S and f:
                    # diffusion matrix
                    DM = S*diag(f)*S.T
                else:
                    error = "'C' cannot be determined "
                    if g:
                        error += "(provide 'C', 'B' or 'DM')"
                    else:
                        error += "(provide 'C', 'B', 'DM' or 'S' and 'f')"
                    raise DefinitionError(error)
            B = DM.cholesky()

        # set existing variables as class attributes
        if phis:
            self.phis = Matrix(phis)
        if phi:
            self.phi = Matrix(phi)
        if eta:
            self.eta = Matrix(eta)
        if S:
            self.S  = Matrix(S)
        if f:
            self.f  = Matrix(f)
        if g:
            self.g  = Matrix(g)
        if A:
            self.A  = Matrix(A)
        if C:
            self.C  = Matrix(C)
        if B:
            self.B  = Matrix(B)
        if DM:
            self.DM = Matrix(DM)

        # attempt to calculate C directly
        # definition: diploma thesis equation (3.30)
        if self.C_attempt:
            self.C = _lyapunov_equation_C(self, self.A, self.B)
            if self.C:
                self.C = matsimp(self.C)
        else:
            # instead the Taylor coefficients will be calculated if needed
            self.C = None
        if verbose:
            self.print_out("ReactionSystem", verbose)

    #-----------------------------------------
    @staticmethod
    def from_string(data=None, yaml_file=None, C_attempt=False, verbose=0):
        """
        Create object from strings in a dictionary or file.
        
        :Parameters:
            - `data`: dictionary of strings
            - `yaml_file`: yaml file defining a dictionary of strings
            - `verbose`: print the obtained system definition (0: not at all,
                    or with 1: print, 2: sympy pprint, 3: IPython display)

        The keys 'concentrations', 'extrinsic_variables', 'transition_rates'
        and 'stoichiometric_matrix' are required and may in part defined by
        'data' and by 'yaml_file'.  To choose non-default symbols, there are
        the optional keys 'normal_variances', 'inverse_correlation_times',
        'frequency' and 'system_size'.

        :Returns: ReactionSystem object

        :Example:
            see module docstring
        """
        data = string_parser(data=data, yaml_file=yaml_file, verbose=verbose)
        return ReactionSystem(data, C_attempt=C_attempt, verbose=verbose)

    #-----------------------------------------
    def copy(self):
        """
        Returns a clone of self.
        """
        data = {'eta': self.eta, 'A': self.A, 'B': self.B}
        new = ReactionSystem(data, map_dict=self.map_dict)
        for key, item in self.__dict__.items():
            setattr(new, key, item)
        return new

    #-----------------------------------------
    def eval_at_phis(self, phis=None, solver=None, select=None,
            C_attempt=False, verbose=0):
        """
        Substitute concentrations phi by phis.
        A warning is printed if g(phis) is not zero.

        :Warning:
            Be sure, that phis is the correct stationary state!

        :Parameters:
            - `phis`: Macroscopic stationary state concentrations
                    If not defined here, self.phis will be used instead.
            - `solver`: solver, phis = solver(self.g, self.phi, select)
                    or phis = solver(self.g, self.phi), e.g. sympy.solve
            - `select`: selected solution from solver (may be ambiguous)
            - `C_attempt`: If True, the calculation of C is attempted.
                    This is forced, if the object was created with the
                    option C_attempt=True.  The calculation is in general
                    not possible and may be unnecessarily time consuming.
            - `verbose`: print phis, A, B, DM and C (0: not at all, or
                    with 1: print, 2: sympy pprint, 3: IPython display)
        """
        if solver:
            if self.phis:
                raise DefinitionError("The system has already been "
                        +"evaluated at stationary state.")
            def sympy_wrapper():
                """for compatibility with sympy.solve"""
                try:
                    phis = solver(self.g, self.phi)
                except SympifyError:
                    phis = solver(self.g, list(self.phi))
                if isinstance(phis, dict):
                    phis = self.phi.subs(phis)
                elif isinstance(phis, list) and phis:
                    try:
                        phis = phis[select]
                    except IndexError:
                        phis = phis[0]
                    except TypeError:
                        phis = phis[0]
                return phis
            try:
                phis = solver(self.g, self.phi, select)
            except AttributeError:
                phis = sympy_wrapper()
            except SympifyError:
                phis = sympy_wrapper()
            if not phis:
                raise DefinitionError("The equation 'g(phi) = 0' could "
                        +"not be solved by '%s'." % solver.__name__)
        if phis:
            self.phis = Matrix(phis)
        if self.phi and self.phis:
            for label in SYMBOLS:
                # set back memoization
                setattr(self, "memo_"+label, {})
            for label in self._EVAL:
                try:
                    for j, phij in enumerate(self.phi):
                        matrix = getattr(self, label)
                        matrix = matsimp(matrix.subs({phij: self.phis[j]}),
                                self.omega)
                        setattr(self, label, matrix)
                except AttributeError:
                    pass
        else:
            raise NameError("The stationary state 'phis' is not defined.")

        if C_attempt or self.C_attempt:
            # attempt to calculate C directly
            # definition: diploma thesis equation (3.30)
            self.C = _lyapunov_equation_C(self, self.A, self.B)
            if self.C:
                self.C = matsimp(self.C)
                self.C_attempt = True

        # check if g is zero
        if self.g.norm() != 0:
            print("WARNING: g seems not to be zero in stationary state.")
            verbose = 0
        if verbose:
            self.print_out("eval_at_phis", verbose)

    #-----------------------------------------
    def num_eval(self, num_dict, map_dict=True, C_attempt=False,
            ifevalf=True, verbose=0):
        """
        Numerical evaluation.

        Evaluate 'f', 'g', 'A', 'B', 'DM' and, if they exist,
        'C' and 'phis', at the values given by 'num_dict'.

        :Parameters:
            - `num_dict`: dictionary with substitutions, e.g. {'k': 5} or {k: 5}
            - `map_dict`: optional dictionary to map string num_dict-keys to
                    symbols; set to False or None to avoid using self.map_dict
                    which is automatically created by self.from_string())
            - `C_attempt`: If True, the calculation of C is attempted.
                    This is forced, if the ReactionSystem was called with
                    the option C_attempt=True.  The calculation is in general
                    not possible and may be unnecessarily time consuming.
            - `ifevalf`: apply .evalf() after numerical substitution or not
            - `verbose`: print phis, A, B, DM and C (0: not at all, or
                    with 1: print, 2: sympy pprint, 3: IPython display)

        :Example: see module docstring
        """
        N = def_N(ifevalf)
        if num_dict != {}:
            # set back memoization
            for label in SYMBOLS:
                setattr(self, "memo_"+label, {})
            self.TH_kernel = None
        # substitution dictionary
        if not map_dict and not isinstance(map_dict, dict):
            subs_dict = num_dict
        elif map_dict == True and not self.map_dict:
            subs_dict = num_dict
        else:
            if map_dict == True:
                map_dict = self.map_dict
            elif self.map_dict:
                new_dict = self.map_dict.copy()
                new_dict.update(map_dict)
                map_dict = new_dict.copy()
                del(new_dict)
            subs_dict = {}
            for key, item in num_dict.items():
                try:
                    subs_dict.update({map_dict[key]: item})
                except KeyError as args:
                    raise KeyError("Please add '%s' to 'map_dict'." % args)
        self.f = matsimp(N(self.f.subs(subs_dict)))
        self.g = matsimp(N(self.g.subs(subs_dict)))
        self.A = matsimp(N(self.A.subs(subs_dict)))
        self.B = matsimp(N(self.B.subs(subs_dict)))
        self.DM = matsimp(N(self.DM.subs(subs_dict)))
        self.etavars = N(self.etavars.subs(subs_dict))
        self.etaKs   = N(self.etaKs.subs(subs_dict))
        self.Omega   = N(self.Omega.subs(subs_dict))
        if self.phis:
            self.phis = N(self.phis.subs(subs_dict))
        if C_attempt or self.C_attempt:
            self.C = _lyapunov_equation_C(self, self.A, self.B)
            if self.C:
                self.C = matsimp(N(self.C))
                self.C_attempt = True
        if verbose:
            self.print_out("num_eval", verbose)

    #-----------------------------------------
    def check_eigenvalues(self, verbose=True):
        """
        Print and return the eigenvalues of the Jacobian A(eta=0) and
        raise EigenvalueError if eigenvalues have a positive real part.
        """
        eigenvalues = []
        for value, mult in self.eval_symbol('A', ()).eigenvals().items():
            eigenvalues += [value]*mult
        if verbose:
            print("The eigenvalues of the Jacobian A are %s" % eigenvalues)
            print("for vanishing extrinsic fluctuations.")
        pos_real_part = False
        try:
            pos_real_part = list(filter(lambda i:
                sympify(i).as_real_imag()[0].is_positive, eigenvalues))
        except TypeError:
            print("WARNING: cannot check if the real part of the "
                    +"eigenvalues of A is negative")
        if pos_real_part:
            raise EigenvalueError("The Jacobian must have eigenvalues "
                    +"with negative real part only: %s." % pos_real_part)
        return eigenvalues

    #-----------------------------------------
    def print_out(self, header=None, verbose=1):
        """
        Print phis, A, B, DM and, if existing, C.

        :Parameters:
            - `header`: optionally print a header
            - `verbose`: use different print functions (1: use print,
                    2: use sympy pprint, 3: use IPython display)
        """
        nprint = def_nprint(verbose, indent=4)
        if header:
            print("=== %s ===" % header)
        if self.phis:
            print("phis =")
            nprint(self.phis)
        if self.g.norm() != 0:
            print("g =")
            nprint(self.g)
        print("A =")
        nprint(self.A)
        print("B =")
        nprint(self.B)
        print("DM =")
        nprint(self.DM)
        if self.C:
            print("C =")
            nprint(self.C)


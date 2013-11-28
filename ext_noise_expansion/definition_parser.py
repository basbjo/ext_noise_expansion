#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Convert strings that define a reaction system to sympy objects.
"""

from sympy import Symbol, Matrix
from sympy.parsing.sympy_parser import parse_expr
from sympy import Add, Mul, Pow, Number
from .tools_objects import DefinitionError
from .tools_sympy import def_nprint

#-----------------------------------------
def _str_exp_parser(str_list):
    """
    Parse strings in str_list and return Matrix with Symbols.  Each symbol
    may be followed by an exponent.  If str_list is empty return None.
    """
    if not str_list:
        return None
    new_list = []
    for expr in str_list:
        (base, exp) = parse_expr(expr).as_base_exp()
        new_list.append(Symbol(str(base), positive=True)**exp)
    return Matrix(new_list)

#-----------------------------------------
def _all_symbols(expr_list, symbol_list=None):
    """
    Return a list of all symbols in a list of sympy expressions.
    Additional symbols can be appended by means of symbol_list.
    """
    new_list = []
    if not symbol_list:
        symbol_list = []
    for expr in expr_list:
        if isinstance(expr, Symbol):
            symbol_list.append(expr)
        elif isinstance(expr, Add):
            new_list.extend(expr.as_ordered_terms())
        elif isinstance(expr, Mul):
            new_list.extend(expr.as_ordered_factors())
        elif isinstance(expr, Pow):
            new_list.extend(expr.as_base_exp())
        elif isinstance(expr, Number):
            pass
        else:
            raise NotImplementedError("Unrecognized expression in 'f'.")
    if new_list == []:
        return list(set(symbol_list))
    else:
        return _all_symbols(new_list, symbol_list)

#-----------------------------------------
def _print_out(data, func=None, pretty=False):
    """
    Print information that has been obtained.
    """
    nprint = def_nprint(pretty, indent=4)
    if func:
        print("=== %s ===" % func)
    print("The chemical network consists of")
    print("%6d component[s]," % data['N'])
    print("%6d reactions and" % data['R'])
    print("%6d extrinsic stochastic variable[s]." % data['M'])
    if pretty:
        print('')
    print('Concentrations of the components:')
    nprint(data['phi'])
    print('Extrinsic stochastic variables:')
    nprint(data['eta'])
    print('Variances (normal distribution):')
    nprint(data['etavars'])
    print('Inverse correlation times:')
    nprint(data['etaKs'])
    print('Stoichiometric matrix:')
    nprint(data['S'])
    print('Macroscopic transition rates:')
    nprint(data['f'])

#-----------------------------------------
def string_parser(data=None, yaml_file=None, verbose=False, pretty=False):
    """
    Generate a data dictionary for the ReactionSystem class from defining
    strings.  ReactionSystem.from_string() uses this function.
    
    :Parameters:
        - `data`: dictionary of strings
        - `yaml_file`: yaml file defining a dictionary of strings
        - `verbose`: print the obtained system definition
        - `pretty`: use pprint instead of print

    The keys 'concentrations', 'extrinsic_variables', 'transition_rates'
    and 'stoichiometric_matrix' are required and may in part be defined by
    'data' and by 'yaml_file'.  To choose non-default symbols, there are
    the optional keys 'normal_variances', 'inverse_correlation_times',
    'frequency' and 'system_size'.

    :Returns:
        Dictionary with the following keys: 'phi', 'f', 'S', 'eta',
        'etaKs', 'etavars' (sympy matrices) and 'map_dict' (dict).

    :Example:

        With a dictionary:
            >>> data = {'concentrations': ['phi'],
            ...         'extrinsic_variables': ['eta'],
            ...         'normal_variances': ['epsilon**2'],
            ...         'inverse_correlation_times': ['K'],
            ...         'stoichiometric_matrix': [[1, -1]],
            ...         'transition_rates': ['l', 'k*(1 + eta)*phi'],
            ...         'frequency': 'omega', 'system_size': 'Omega'}
            >>> data = string_parser(data, verbose=True)
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
                [epsilon**2]
            Inverse correlation times:
                [K]
            Stoichiometric matrix:
                [1, -1]
            Macroscopic transition rates:
                [l]
                [k*phi*(eta + 1)]

        With a yaml file:
            >>> data2 = string_parser(None, 'test_system.yaml')
            >>> data == data2
            True
            >>> list(data2.keys()) #doctest: +NORMALIZE_WHITESPACE
            ['phi', 'etaKs', 'omega', 'f', 'map_dict', 'S',
                    'eta', 'etavars', 'Omega']

        ReactionSystem:
            >>> from ext_noise_expansion import ReactionSystem
            >>> rs = ReactionSystem(data2)
    """
    if not data:
        data = {}
    imported_yaml = False
    if yaml_file:
        if not imported_yaml:
            import yaml
            imported_yaml = True
        data.update(yaml.load(open(yaml_file,'r')))

    # get strings of necessary keys
    try:
        phi = data['concentrations']
        eta = data['extrinsic_variables']
        S   = data['stoichiometric_matrix']
        f   = data['transition_rates']
    except KeyError as arg:
        raise DefinitionError("Key %s is missing " % arg
                +"in the system defintion.")
    # optional:
    etavars = data.get('normal_variances')
    etaKs   = data.get('inverse_correlation_times')
    omega   = data.get('frequency')
    Omega   = data.get('system_size')

    # determine number of components, extrinsic variables and reactions
    N = len(phi)
    M = len(eta)
    R = len(f)
    if (etavars and M != len(etavars)) or (etaKs and M != len(etaKs)):
        raise DefinitionError("'extrinsic_variables', 'normal_variances' and "
                +"'inverse_correlation_times' must have the same length.")
    if N != len(S) or R != len(S[0]) or len(set([len(i) for i in S])) != 1:
        raise DefinitionError("The shape of 'stoichiometric_matrix' is "
                +"incompatible to 'concentrations' or 'transition_rates'.")

    newdata = {}
    # build matrices of symbols (symbols may be followed by an exponent)
    newdata['phi']     = _str_exp_parser(phi)
    newdata['eta']     = _str_exp_parser(eta)
    # optional
    newdata['etavars'] = _str_exp_parser(etavars)
    newdata['etaKs']   = _str_exp_parser(etaKs)

    # optional symbols and initialize list of all symbols
    symbols = []
    if omega:
        symbols.append(Symbol(omega, positive=True))
        newdata['omega'] = symbols[-1]
    if Omega:
        symbols.append(Symbol(Omega, positive=True))
        newdata['Omega'] = symbols[-1]

    # build stoichiometric matrix and transition rates
    newdata['S'] = Matrix(S)
    f = [parse_expr(i) for i in f]
    symbols += _all_symbols(f)
    pos_symbols = [] # duplicates with positive=True
    to_pos_dict = {} # maps symbols to duplicates
    map_dict = {} # maps strings to duplicates
    for symbol in symbols:
        string = str(symbol)
        new_symbol = Symbol(string, positive=True)
        pos_symbols.append(new_symbol)
        to_pos_dict.update({symbol: new_symbol})
        map_dict.update({string: new_symbol})
    f = Matrix([i.subs(to_pos_dict) for i in f])

    # build dictionary and print out if verbose=True
    newdata.update({'f': f, 'map_dict': map_dict})
    verbosedata = newdata.copy()
    verbosedata.update({'N': N, 'M': M, 'R': R})
    if verbose or pretty:
        _print_out(verbosedata, 'string_parser', pretty)
    return newdata


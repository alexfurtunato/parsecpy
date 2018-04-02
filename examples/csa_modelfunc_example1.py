#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Model function example to use with parsecpy runmodel script.

    Speedup:
        S = 1 / ( ( 1-f(p,N) ) + f(p,N)/p + Q(p,N) )

    Parallel Fraction:
        f(p,N) = max( min((f1) + (f2)/p + (f3)*(f4)^N,1 ),0 )

    Overhead:
        Q(p,N) = (f5) + ( (f6)*p )/( (f7)^N )

"""

import random
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def get_parallelfraction(param, args):
    """
    Return a Dataframe with parallel fraction calculate of model predicted.

    :param param: Parameters of model overhead part
    :param args: Positional arguments passed for objective
                 and constraint functions
    """

    pf = pd.DataFrame()
    for sizename, n in args[2]:
        pf[sizename] = _func_parallelfraction(param[:4], args[1], n)
    pf.set_index(args[1], inplace=True)
    return pf


def get_overhead(param, args):
    """
    Return a Dataframe with overhead calculate with of model predicted.

    :param param: Parameters of model overhead part
    :param args: Positional arguments passed for objective
                 and constraint functions
    """

    oh = pd.DataFrame()
    for sizename, n in args[2]:
        oh[sizename] = _func_overhead(param[:4], args[1], n)
    oh.set_index(args[1], inplace=True)
    return oh


def _func_parallelfraction(f, p, n):
    """
    Model function that calculate the parallel fraction.

    :param f: Parameters of model
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    """

    return f[0] + f[1]/p + f[2]*pow(f[3], n)


def _func_overhead(q, p, n):
    """
    Model function that calculate the overhead.

    :param q: Parameters of model overhead part
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    """

    return q[0]+(q[1]*p)/pow(q[2], n)


def _func_speedup(fparam, p, n):
    """
    Model function that calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    """

    f = _func_parallelfraction(fparam, p, n)
    return 1/((1-f)+f/p)


def _func_speedup_with_overhead(fparam, p, n):
    """
    Model function that calculate the speedup with overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    """

    f = _func_parallelfraction(fparam[:4], p, n)
    q = _func_overhead(fparam[4:], p, n)
    return 1/((1-f)+f/p+q)


def model(p, args):
    """
    Model function that represent the mathematical model used
    to predict parameters.

    :param p: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    """

    y_pred = pd.DataFrame()
    for sizename, n in args[2]:
        if args[0]:
            param = p
            modelserie = _func_speedup_with_overhead(param, args[1], n)
        else:
            param = p[:4]
            modelserie = _func_speedup(param, args[1], n)
        y_pred[sizename] = modelserie
    y_pred.set_index(args[1], inplace=True)
    return y_pred


def probe_function(p, tgen, *args):
    """
    Constraint function that would be considered on model.

    :param p: Actual parameters values
    :param tgen: Temperature of generation
    :param args: Positional arguments passed for objective
                 and constraint functions
    """

    probe_solution = []
    for i, x in enumerate(p):
        w = random.uniform(0, 1)
        r = math.tan(math.pi*(w - 0.5))
        ps = 2*np.mod((x + r*tgen + 1)/2, 1)-1
        probe_solution.append(ps)
    return probe_solution


def objective_function(p, *args):
    """
    Objective function (target function) to minimize.

    :param p: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    """

    y_measure = args[0]
    args = args[1:]
    y_pred = model(p, args)
    return mean_squared_error(y_measure, y_pred)

# -*- coding: utf-8 -*-
"""
    Model function example to use with parsecpy runmodel script.

    Speedup:
        S = 1 / ( ( 1-f(p,n) ) + f(p,n)/p + Q(p,n) )

    Parallel Fraction:
        f(p,n) = max( min((f1) + (f2)/p + (f3)*(f4)^n,1 ),0 )

    Overhead:
        Q(p,n) = (f5) + ( (f6)*p )/( (f7)^n )

"""

import random
import numpy as np
import math
from sklearn.metrics import mean_squared_error


def get_parallelfraction(param, x):
    """
    Get the calculated parallel fraction of model.

    :param param: Actual parameters values
    :param x: Inputs array
    :return: Tuple with input array and predicted parallel fraction array
    """

    pf = []
    for p, n in x:
        pf.append(_func_parallelfraction(param[:4], p, n))
    return x, pf


def get_overhead(param, x):
    """
    Get the calculated overhead of model.

    :param param: Actual parameters values
    :param x: Inputs array
    :return: Tuple with input array and predicted overhead array
    """

    oh = []
    for p, n in x:
        oh.append(_func_overhead(param[:4], p, n))
    return x, oh


def _func_parallelfraction(f, p, n):
    """
    Model function to calculate the parallel fraction.

    :param f: Actual parallel fraction parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated parallel fraction value
    """

    fp = f[0] + f[1]/p + f[2]*pow(f[3], n)
    return max(min(fp, 1), 0)


def _func_overhead(q, p, n):
    """
    Model function to calculate the overhead.

    :param q: Actual overhead parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated overhead value
    """

    return q[0]+(q[1]*p)/pow(q[2], n)


def _func_speedup(fparam, p, n):
    """
    Model function to calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated speedup value
    """

    f = _func_parallelfraction(fparam, p, n)
    return 1/((1-f)+f/p)


def _func_speedup_with_overhead(fparam, p, n):
    """
    Model function to calculate the speedup with overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated speedup value
    """

    f = _func_parallelfraction(fparam[:4], p, n)
    q = _func_overhead(fparam[4:], p, n)
    return 1/((1-f)+f/p+q)


def model(par, x, oh):
    """
    Mathematical Model function to predict the measures values.

    :param par: Actual parameters values
    :param x: inputs array
    :param oh: If should be considered the overhead
    :return: Tuple with input array and predicted output array
    """

    pred = []
    for p, n in x:
        if oh:
            param = par[:]
            y_model = _func_speedup_with_overhead(param, p, n)
        else:
            param = par[:4]
            y_model = _func_speedup(param, p, n)
        pred.append(y_model)
    return x, pred


def probe_function(par, tgen, pxmin, pxmax, *args):
    """
    Constraint function that would be considered on model.

    :param par: Actual parameters values
    :param tgen: Temperature of generation
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: A new probe solution based on tgen and a random function
    """

    probe_solution = []
    limits = True
    if pxmin is None or pxmax is None:
        limits = False
    for i, p in enumerate(par):
        r = random.uniform(0, 1)
        t = math.tan(math.pi*(r - 0.5))
        if limits:
            ps = pxmin[i] + np.mod(p + t*tgen, pxmax[i]-pxmin[i])
        else:
            ps = np.mod(p + t*tgen, 10)
        probe_solution.append(ps)
    return probe_solution


def objective_function(par, *args):
    """
    Objective function (target function) to minimize.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: Mean squared error between measures and predicts
    """

    measure = args[1]
    pred = model(par, measure['x'], args[0])
    return mean_squared_error(measure['y'], pred[1])

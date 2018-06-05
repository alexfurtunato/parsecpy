# -*- coding: utf-8 -*-
"""
    Model function example to use with parsecpy runmodel script.
    Model for variation of input size (n) and number of cores (p).

    Speedup:
        S = 1 / ( ( 1-f(p,n) ) + f(p,n)/p + Q(p,n) )

    Parallel Fraction:
        f(p,n) = max( min((f1) + (f2)/p + (f3)*(f4)^n,1 ),0 )

    Overhead:
        Q(p,n) = (f5) + ( (f6)*p )/( (f7)^n )

"""

import numpy as np
from sklearn.metrics import mean_squared_error

sockets_enable = True
cores_per_socket = 16

def get_parallelfraction(param, x):
    """
    Get the calculated parallel fraction of model.

    :param param: Actual parameters values
    :param x: Inputs array
    :return: Dict with input array ('x') and predicted parallel
             fraction array ('pf')
    """

    pf = []
    for p, n in x:
        pf.append(_func_parallelfraction(param[:4], p, n))
    return {'x': x, 'pf': pf}


def get_overhead(param, x):
    """
    Get the calculated overhead of model.

    :param param: Actual parameters values
    :param x: Inputs array
    :return: Dict with input array ('x') and predicted overhead array ('oh')
    """

    oh = []
    for p, n in x:
        oh.append(_func_overhead(param[:4], p, n))
    return {'x': x, 'oh': oh}


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


def _func_speedup(param, p, n, oh):
    """
    Model function to calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param n: Problems size used on model data
    :return: calculated speedup value
    """

    if oh:
        f = _func_parallelfraction(param[:4], p, n)
        q = _func_overhead(param[4:], p, n)
    else:
        f = _func_parallelfraction(param[:], p, n)
        q = 0
    speedup = 1 / ((1 - f) + f / p + q)
    if sockets_enable:
        s = p // (cores_per_socket + 1)
        speedup = speedup - param[-1] * s
    return speedup


def model(par, x, oh):
    """
    Mathematical Model function to predict the measures values.

    :param par: Actual parameters values
    :param x: inputs array
    :param oh: If should be considered the overhead
    :return: Dict with input array ('x') and predicted output array ('y')
    """

    pred = []
    for n, p in x:
        y_model = _func_speedup(par, p, n, oh)
        pred.append(y_model)
    return {'x': x, 'y': pred}


def probe_function(par, tgen):
    """
    Constraint function that would be considered on model.

    :param par: Actual parameters values
    :param tgen: Temperature of generation
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: A new probe solution based on tgen and a random function
    """

    t = np.tan(np.pi * (np.random.uniform(size=len(par))-0.5))
    probe_solution = 2*np.mod((par + t * tgen + 1)/2, 1) - 1
    return probe_solution


def constraint_function(par, x_meas, **kwargs):
    """
    Constraint function that would be considered on model.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: If parameters are acceptable based on return functions
    """

    pred = model(par, x_meas, kwargs['oh'])
    y_min = np.min(pred['y'])
    f_pred = get_parallelfraction(par, x_meas)
    f_max = np.max(f_pred['pf'])
    f_min = np.min(f_pred['pf'])
    is_feasable = np.all(np.array([1-f_max, f_min, y_min]) >= 0)
    return is_feasable


def objective_function(par, x_meas, y_meas, **kwargs):
    """
    Objective function (target function) to minimize.

    :param par: Actual parameters values
    :param args: Positional arguments passed for objective
                 and constraint functions
    :return: Mean squared error between measures and predicts
    """

    pred = model(par, x_meas, kwargs['oh'])
    return mean_squared_error(y_meas, pred['y'])

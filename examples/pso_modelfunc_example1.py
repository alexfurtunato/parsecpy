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

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def get_parallelfraction(param,args):
    """
    Return a Dataframe with parallel fraction calculate of model predicted.

    :param param: Parameters of model overhead part
    :param args: Positional arguments passed for objective and constraint functions
    """

    pf = pd.DataFrame()
    for sizename,N in args[2]:
        pf[sizename] = _func_parallelfraction(param[:4],args[1],N)
    pf.set_index(args[1],inplace=True)
    return pf

def get_overhead(param,args):
    """
    Return a Dataframe with overhead calculate with of model predicted.

    :param param: Parameters of model overhead part
    :param args: Positional arguments passed for objective and constraint functions
    """

    oh = pd.DataFrame()
    for sizename,N in args[2]:
        oh[sizename] = _func_overhead(param[:4],args[1],N)
    oh.set_index(args[1],inplace=True)
    return oh

def _func_parallelfraction(f, p, N):
    """
    Model function that calculate the parallel fraction.

    :param f: Parameters of model
    :param p: Numbers of cores used on model data
    :param N: Problems size used on model data
    """

    return f[0] + f[1]/p + f[2]*pow(f[3],N)

def _func_overhead(q, p, N):
    """
    Model function that calculate the overhead.

    :param q: Parameters of model overhead part
    :param p: Numbers of cores used on model data
    :param N: Problems size used on model data
    """

    return q[0]+(q[1]*p)/pow(q[2],N)

def _func_speedup(fparam, p, N):
    """
    Model function that calculate the speedup without overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param N: Problems size used on model data
    """

    f = _func_parallelfraction(fparam,p,N)
    return 1/((1-f)+f/p)

def _func_speedup_with_overhead(fparam, p, N):
    """
    Model function that calculate the speedup with overhead.

    :param fparam: Actual parameters values
    :param p: Numbers of cores used on model data
    :param N: Problems size used on model data
    """

    f = _func_parallelfraction(fparam[:4],p,N)
    q = _func_overhead(fparam[4:],p,N)
    return 1/((1-f)+f/p+q)

def model(p, args):
    """
    Model function that represent the mathematical model used to predict parameters.

    :param p: Actual parameters values
    :param args: Positional arguments passed for objective and constraint functions
    """

    y_pred = pd.DataFrame()
    for sizename,N in args[2]:
        if args[0]:
            param = p.pos.copy()
            modelserie = _func_speedup_with_overhead(param, args[1], N)
        else:
            param = p.pos[:4]
            modelserie = _func_speedup(param, args[1], N)
        y_pred[sizename] = modelserie
    y_pred.set_index(args[1], inplace=True)
    return y_pred

def constraint_function(p, *args):
    """
    Constraint function that would be considered on model.

    :param p: Actual parameters values
    :param args: Positional arguments passed for objective and constraint functions
    """

    args = args[1:]
    y_pred = model(p, args)
    y_min = y_pred.min().min()
    f_pred = get_parallelfraction(p.pos,args)
    f_max = f_pred.max().max()
    f_min = f_pred.min().min()
    is_feasable = np.all(np.array([1-f_max,f_min,y_min])>=0)
    return is_feasable

def objective_function(p, *args):
    """
    Objective function (target function) to minimize.

    :param p: Actual parameters values
    :param args: Positional arguments passed for objective and constraint functions
    """

    y_measure = args[0]
    args = args[1:]
    y_pred = model(p, args)
    return mean_squared_error(y_measure, y_pred)

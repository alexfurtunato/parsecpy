#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel [-h] [-f PARSECPYFILENAME] [-o OVERHEAD]
                         [-x MAXITERATIONS] [-p PARTICLES] [-t THREADS]
                         [-m MODELFILEABSPATH]

    Script to run swarm modelling to predict aparsec application output

    optional arguments:
        -h, --help            show this help message and exit
        -f PARSECPYFILENAME, --parsecpyfilename PARSECPYFILENAME
                        Run output filename from Parsec specificated package.
        -o OVERHEAD, --overhead OVERHEAD
                        If it consider the overhead on model function
        -d DIMENSION, --dimension DIMENSION
                        Number of parameters of model
        -s STEPS, --steps STEPS
                        Number of steps to run
        -a ANNEALERS, --annealers ANNEALERS
                        Number of annealers to use on model
        -t THREADS, --threads THREADS
                        Number of Threads to run the algorithm
        -m MODELFILEABSPATH, --modelfileabspath MODELFILEABSPATH
                        Absolute path from Python file with model function.
        -v VERBOSE, --verbose VERBOSE
                        Verbose level: 0, 1 or 2.

    Example
        parsecpy_runmodel_csa -f /var/myparsecsim.dat -m /var/mymodelfunc.py -d 5 -s 1000 -a 10
"""

import os
import sys
import time
import random
import argparse
from parsecpy import ParsecData
from parsecpy import CoupledAnnealer

try:
    xrange
except NameError:
    xrange = range

def argsparsefloatlist(txt):
    """
    Validate the list int argument.

    :param txt: argument of comma separated int strings.
    :return: list of integer converted ints.
    """

    txt = txt.split(',')
    listarg = [float(i.strip()) for i in txt]
    return listarg

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run csa '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-f','--parsecpyfilename', required=True,
                        help='Input filename from Parsec specificated package.')
    parser.add_argument('-o','--overhead', type=bool,
                        help='If it consider the overhead', default=False)
    parser.add_argument('-d','--dimension', type=int,
                        help='Number of parameters', default=5)
    parser.add_argument('-s','--steps', type=int,
                        help='Number max of iterations', default=100)
    parser.add_argument('-a','--annealers', type=int,
                        help='Number of annealers', default=10)
    parser.add_argument('-t','--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-m','--modelfileabspath', required=True,
                        help='Absolute path from Python file with the'
                             'objective function.')
    parser.add_argument('-v','--verbose', type=int,
                        help='If it shows output verbosily: Values: 0, 1, 2', default=1)
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    print("Processing the Model")

    args = argsparsevalidation()

    if os.path.isfile(args.modelfileabspath):
        modelpath = args.modelfileabspath
    else:
        print('Error: You should inform the correct module of objective function to model')

    initial_state = [
           tuple((random.normalvariate(0, 5) for _ in xrange(args.dimension)))
           for x in xrange(args.annealers)]

    parsec_exec = ParsecData(args.parsecpyfilename)
    y_measure = parsec_exec.speedups()
    N = [(col, int(col.split('_')[1])) for col in y_measure]
    p = y_measure.index
    argsanneal = (y_measure, args.overhead, p, N)

    A = CoupledAnnealer(
        modelpath,
        n_annealers=args.annealers,
        initial_state=initial_state,
        tgen_initial=0.01,
        tacc_initial=0.1,
        steps=args.steps,
        processes=args.threads,
        verbose=args.verbose,
        update_interval=100,
        args=argsanneal
    )

    repititions = range(10)
    err_min = 0

    starttime = time.time()
    for i in repititions:
        print('Iteration: ',i+1)
        model = A.run()
        if i == 0:
            err_min = model.error
            print('Error: ', err_min)
        else:
            if model.error < err_min:
                print('Error: ', err_min, '->', model.error)
                err_min = model.error
        endtime = time.time()
        print('Execution time = %s seconds' % (endtime - starttime))
        starttime = endtime

    print('Best Params: \n',model.params)
    print('Measured: \n',y_measure)
    print('Model: \n',model.y_model)

    model.savedata(parsec_exec.config)
    print('Terminado!')

if __name__ == '__main__':
    main()
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
        -x MAXITERATIONS, --maxiterations MAXITERATIONS
                        Number max of iterations to run the algorithm
        -p PARTICLES, --particles PARTICLES
                        Number of particles used on pso
        -t THREADS, --threads THREADS
                        Number of Threads to run the algorithm
        -m MODELFILEABSPATH, --modelfileabspath MODELFILEABSPATH
                        Absolute path from Python file with model function.
    Example
        parsecpy_runprocess -p frqmine -c gcc-hooks -r 5 -i native 1,2,4,8
"""

import os
import sys
import time
import argparse
from parsecpy import ParsecData
from parsecpy import Swarm

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

    parser = argparse.ArgumentParser(description='Script to run swarm '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-f','--parsecpyfilename', required=True,
                        help='Input filename from Parsec specificated package.')
    parser.add_argument('-l', '--lowervalues', type=argsparsefloatlist, required=True,
                        help='List of minimum particles values '
                             'used. Ex: -1,0,-2,0')
    parser.add_argument('-u', '--uppervalues', type=argsparsefloatlist, required=True,
                        help='List of maximum particles values '
                             'used. Ex: 5,2,1,10')
    parser.add_argument('-o','--overhead', type=bool,
                        help='If it consider the overhead', default=False)
    parser.add_argument('-x','--maxiterations', type=int,
                        help='Number max of iterations', default=100)
    parser.add_argument('-p','--particles', type=int,
                        help='Number of particles', default=100)
    parser.add_argument('-t','--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-m','--modelfileabspath', required=True,
                        help='Absolute path from Python file with the'
                             'objective function.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    print("Processing the Model")

    # adjust list of arguments to avoid negative number values error
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg

    args = argsparsevalidation()

    if os.path.isfile(args.modelfileabspath):
        modelpath = args.modelfileabspath
    else:
        print('Error: You should inform the correct module of objective function to model')

    l = args.lowervalues
    u = args.uppervalues

    parsec_exec = ParsecData(args.parsecpyfilename)
    y_measure = parsec_exec.speedups()
    N = [(col, int(col.split('_')[1])) for col in y_measure]
    p = y_measure.index
    argsswarm = (y_measure, args.overhead, p, N)

    S = Swarm(l, u, args=argsswarm, threads=args.threads, size=args.particles,
              maxiter=args.maxiterations, modelpath=modelpath)

    repititions = range(10)
    err_min = 0

    starttime = time.time()
    for i in repititions:
        print('Iteration: ',i+1)
        model = S.run()
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
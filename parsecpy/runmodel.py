#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a parsec application.

    Its possible use a loop to repeat the same single parsec application
    run on specific number of times; And, also, its possible to refer
    differents input sizes to generate executions and resume all on a
    Pandas Dataframe with times and speedups.

    parsecpy_runprocess [-h] -p PACKAGE
        [-c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}]
        [-i INPUT] [-r REPITITIONS] c

    Script to run parsec app with repetitions and multiples inputs sizes

    positional arguments
        c
            List of cores numbers to be used. Ex: 1,2,4

    optional arguments
        -h, --help
            show this help message and exit
        -p PACKAGE, --package PACKAGE
            Package Name to run
        -c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb},
            --compiler {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}
            Compiler name to be used on run. (Default: gcc-hooks).
        -i INPUT, --input INPUT
            Input name to be used on run. (Default: native).
            Syntax: inputsetname[<initialnumber>:<finalnumber>]. Ex: native or native_1:10
        -r REPITITIONS, --repititions REPITITIONS
            Number of repititions for a specific run. (Default: 1)
    Example
        parsecpy_runprocess -p frqmine -c gcc-hooks -r 5 -i native 1,2,4,8
"""

import os
import time
import argparse
from parsecpy import ParsecData
from parsecpy import Swarm

def argsparseintlist(txt):
    """
    Validate the list int argument.

    :param txt: argument of comma separated int strings.
    :return: list of integer converted ints.
    """

    print('TXT: ',txt)
    txt = txt.split(',')
    listarg = [int(i) for i in txt]
    return listarg

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run swarm '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-f','--parsecpyfilename', help='Input filename from '
                                                        'Parsec specificated '
                                                        'package.')
    parser.add_argument('-o','--overhead', type=bool,
                        help='If it consider the overhead', default=False)
    parser.add_argument('-x','--maxiterations', type=int,
                        help='Number max of iterations', default=100)
    parser.add_argument('-p','--particles', type=int,
                        help='Number of particles', default=100)
    parser.add_argument('-t','--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-m','--modelfileabspath', help='Absolute path from '
                                                        'Python file with the'
                                                        'objective function.')
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

    if(args.overhead):
        print('Com Overhead:')
        l = [-10,-10,-10,-10,-10,-10,-10]
        u = [10,10,10,10,10,10,10]
    else:
        l = [-10,-10,-10,-10]
        u = [10,10,10,10]

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
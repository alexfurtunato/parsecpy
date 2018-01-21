#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a csa model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel_csa [-h] -f PARSECPYFILENAME [-o OVERHEAD]
                                 [-d DIMENSION] [-s STEPS] [-u UPDATE_INTERVAL]
                                 [-a ANNEALERS] [-t THREADS] [-r REPETITIONS] -m
                                 MODELFILEABSPATH [-v VERBOSE]

    Script to run csa modelling to predict aparsec application output

    optional arguments:
      -h, --help            show this help message and exit
      -f PARSECPYFILENAME, --parsecpyfilename PARSECPYFILENAME
                            Input filename from Parsec specificated package.
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -d DIMENSION, --dimension DIMENSION
                            Number of parameters
      -s STEPS, --steps STEPS
                            Number max of iterations
      -u UPDATE_INTERVAL, --update_interval UPDATE_INTERVAL
                            Number steps to run cooling temperatures
      -a ANNEALERS, --annealers ANNEALERS
                            Number of annealers
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to algorithm execution
      -m MODELFILEABSPATH, --modelfileabspath MODELFILEABSPATH
                            Absolute path from Python file with theobjective
                            function.
      -v VERBOSE, --verbose VERBOSE
                            If it shows output verbosily: Values: 0, 1, 2

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
    parser.add_argument('-u','--update_interval', type=int,
                        help='Number steps to run cooling temperatures', default=1)
    parser.add_argument('-a','--annealers', type=int,
                        help='Number of annealers', default=10)
    parser.add_argument('-t','--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-r','--repetitions', type=int,
                        help='Number of repetitions to algorithm execution', default=10)
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


    parsec_exec = ParsecData(args.parsecpyfilename)
    y_measure = parsec_exec.speedups()
    N = [(col, int(col.split('_')[1])) for col in y_measure]
    p = y_measure.index
    argsanneal = (y_measure, args.overhead, p, N)

    repititions = range(args.repetitions)
    err_min = 0
    computed_models = []
    best_model_idx = 0

    starttime = time.time()
    for i in repititions:
        print('Iteration: ',i+1)

        initial_state = [
            tuple((random.normalvariate(0,5) for _ in xrange(args.dimension)))
            for x in xrange(args.annealers)]

        A = CoupledAnnealer(
            modelpath,
            n_annealers=args.annealers,
            initial_state=initial_state,
            tgen_initial=1,
            tacc_initial=0.1,
            steps=args.steps,
            threads=args.threads,
            verbose=args.verbose,
            update_interval=args.update_interval,
            args=argsanneal
        )
        model = A.run()
        computed_models.append(model)
        if i == 0:
            err_min = model.error
            print('Error: ', err_min)
        else:
            if model.error < err_min:
                best_model_idx = i
                print('Error: ', err_min, '->', model.error)
                err_min = model.error
        endtime = time.time()
        print('Execution time = %s seconds' % (endtime - starttime))
        starttime = endtime

    print('\n\n***** Done! *****\n')
    print('Error: %.8f \nPercentual Error (Measured Mean): %.8f %%' % (err_min, 100*err_min/y_measure.mean().mean()))
    print('Best Parameters: \n',computed_models[best_model_idx].params)
    print('\nMeasured Speedups: \n',y_measure)
    print('\nModeled Speedups: \n',computed_models[best_model_idx].y_model)

    fn = computed_models[best_model_idx].savedata(parsec_exec.config)
    print('Model data saved on filename: %s' % fn)

if __name__ == '__main__':
    main()
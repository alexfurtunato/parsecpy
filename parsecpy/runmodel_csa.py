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
                                 [-a ANNEALERS] [-t THREADS] [-r REPETITIONS]
                                 -m MODELFILEABSPATH [-v VERBOSE]

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
        parsecpy_runmodel_csa -f /var/myparsecsim.dat -m /var/mymodelfunc.py
        -d 5 -s 1000 -a 10
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from parsecpy import ParsecData
from parsecpy import CoupledAnnealer


def argsparselist(txt):
    """
    Validate the list of txt argument.

    :param txt: argument of comma separated int strings.
    :return: list of strings.
    """

    txt = txt.split(',')
    listarg = [i.strip() for i in txt]
    return listarg


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
    parser.add_argument('-f', '--parsecpyfilename', required=True,
                        help='Input filename from Parsec '
                             'specificated package.')
    parser.add_argument('-o', '--overhead', type=bool,
                        help='If it consider the overhead', default=False)
    parser.add_argument('-d', '--dimension', type=int,
                        help='Number of parameters', default=5)
    parser.add_argument('-s', '--steps', type=int,
                        help='Number max of iterations', default=100)
    parser.add_argument('-u', '--update_interval', type=int,
                        help='Number steps to run cooling temperatures',
                        default=1)
    parser.add_argument('-a', '--annealers', type=int,
                        help='Number of annealers', default=10)
    parser.add_argument('-n', '--problemsizes', type=argsparselist,
                        help='List of problem sizes to model used. '
                             'Ex: native_01,native_05,native_08')
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions to algorithm execution',
                        default=10)
    parser.add_argument('-m', '--modelfileabspath', required=True,
                        help='Absolute path from Python file with the'
                             'objective function.')
    parser.add_argument('-c', '--crossvalidation', type=bool,
                        help='If run the cross validation of modelling',
                        default=False)
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose', default=0)
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    print("Processing the Model")

    args = argsparsevalidation()

    if os.path.isfile(args.modelfileabspath):
        modelcodepath = args.modelfileabspath
    else:
        print('Error: You should inform the correct module of objective '
              'function to model')
        sys.exit()

    if not os.path.isfile(args.parsecpyfilename):
        print('Error: You should inform the correct parsecpy measures file')
        sys.exit()

    parsec_exec = ParsecData(args.parsecpyfilename)
    y_measure = parsec_exec.speedups()

    n = y_measure.columns
    if args.problemsizes:
        n_model = n.copy()
        for i in args.problemsizes:
            if i not in n_model:
                print('Error: Measures not has especified problem sizes')
                sys.exit()
        y_measure = y_measure[args.problemsizes]

    x = []
    y = []
    for i, row in y_measure.iterrows():
        for c, v in row.iteritems():
            x.append([i, int(c.split('_')[1])])
            y.append(v)
    input_name = c.split('_')[0]
    x = np.array(x)
    y = np.array(y)

    argsanneal = (args.overhead, {'x': x, 'y': y, 'input_name': input_name})

    repetitions = range(args.repetitions)
    err_min = 0
    computed_models = []
    best_model_idx = 0

    starttime = time.time()
    for i in repetitions:
        print('\nAlgorithm Execution: ', i+1)

        initial_state = [
            tuple((random.normalvariate(0, 5) for _ in range(args.dimension)))
            for _ in range(args.annealers)]

        cann = CoupledAnnealer(
            initial_state,
            modelcodepath=modelcodepath,
            n_annealers=args.annealers,
            tgen_initial=0.1,
            tacc_initial=0.9,
            steps=args.steps,
            threads=args.threads,
            verbosity=args.verbosity,
            update_interval=args.update_interval,
            parsecpydatapath=args.parsecpyfilename,
            args=argsanneal
        )
        model = cann.run()
        computed_models.append(model)
        if i == 0:
            err_min = model.error
            print('  Error: %.8f' % err_min)
        else:
            if model.error < err_min:
                best_model_idx = i
                print('  Error: %.8f -> %.8f ' % (err_min,  model.error))
                err_min = model.error
        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))
        starttime = endtime

    print('\n\n***** Modelling Results! *****\n')
    print('Error: %.8f \nPercentual Error (Measured Mean): %.2f %%' %
          (computed_models[best_model_idx].error,
           computed_models[best_model_idx].errorrel))
    if args.verbosity > 0:
        print('Best Parameters: \n', computed_models[best_model_idx].params)
    if args.verbosity > 1:
        print('\nMeasured Speedup: \n', y_measure)
        print('\nModeled Speedup: \n', computed_models[best_model_idx].y_model)

    print('\n***** Modelling Done! *****\n')

    if args.crossvalidation:
        print('\n\n***** Starting cross validation! *****\n')
        starttime = time.time()

        scores = computed_models[best_model_idx].validate(kfolds=10)
        print('\n  Cross Validation (K-fold, K=10) Metrics: ')
        if args.verbosity > 2:
            print('\n   Times: ')
            for key, value in scores['times'].items():
                print('     %s: %.8f' % (key, value.mean()))
                print('     ', value)
        print('\n   Scores: ')
        for key, value in scores['scores'].items():
            if args.verbosity > 1:
                print('     %s: %.8f' % (value['description'],
                                         value['value'].mean()))
                print('     ', value['value'])
            else:
                print('     %s: %.8f' % (value['description'],
                                         value['value'].mean()))

        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))

        print('\n***** Cross Validation Done! *****\n')

    print('\n\n***** ALL DONE! *****\n')

    fn = computed_models[best_model_idx].savedata(parsec_exec.config)
    print('Model data saved on filename: %s' % fn)


if __name__ == '__main__':
    main()

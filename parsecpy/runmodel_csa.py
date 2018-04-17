#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a csa model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel_csa [-h] --config CONFIG -f PARSECPYFILENAME
                             [-a ANNEALERS] [-s STEPS] [-u UPDATE_INTERVAL]
                             [-d DIMENSION] [-n PROBLEMSIZES] [-o OVERHEAD]
                             [-t THREADS] [-r REPETITIONS]
                             [-c CROSSVALIDATION] [-v VERBOSITY]

    Script to run csa modelling to predict aparsec application output

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Filepath from Configuration file configurations
                            parameters
      -f PARSECPYFILENAME, --parsecpyfilename PARSECPYFILENAME
                            Absolute path from Input filename from Parsec
                            specificated package.
      -a ANNEALERS, --annealers ANNEALERS
                            Number of annealers
      -s STEPS, --steps STEPS
                            Number max of iterations
      -u UPDATE_INTERVAL, --update_interval UPDATE_INTERVAL
                            Number steps to run cooling temperatures
      -d DIMENSION, --dimension DIMENSION
                            Number of parameters
      -n PROBLEMSIZES, --problemsizes PROBLEMSIZES
                            List of problem sizes to model used. Ex:
                            native_01,native_05,native_08
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to algorithm execution
      -c CROSSVALIDATION, --crossvalidation CROSSVALIDATION
                            If run the cross validation of modelling
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_csa -f /var/myparsecsim.dat
        --config /var/myconfig.json -d 5 -s 1000 -a 10
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from copy import deepcopy
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
    parser.add_argument('--config', required=True,
                        help='Filepath from Configuration file '
                             'configurations parameters')
    parser.add_argument('-f', '--parsecpyfilepath',
                        help='Absolute path from Input filename from Parsec '
                             'specificated package.')
    parser.add_argument('-a', '--annealers', type=int,
                        help='Number of annealers')
    parser.add_argument('-s', '--steps', type=int,
                        help='Number max of iterations')
    parser.add_argument('-u', '--update_interval', type=int,
                        help='Number steps to run cooling temperatures')
    parser.add_argument('-d', '--dimension', type=int,
                        help='Number of parameters')
    parser.add_argument('-n', '--problemsizes', type=argsparselist,
                        help='List of problem sizes to model used. '
                             'Ex: native_01,native_05,native_08')
    parser.add_argument('-o', '--overhead', type=bool,
                        help='If it consider the overhead')
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of Threads')
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions to algorithm execution')
    parser.add_argument('-c', '--crossvalidation', type=bool,
                        help='If run the cross validation of modelling')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose')
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    print("\n***** Processing the Model *****")

    args = argsparsevalidation()

    if args.config:
        if not os.path.isfile(args.config):
            print('Error: You should inform the correct config file path.')
            sys.exit()
        with open(args.config, 'r') as fconfig:
            config = json.load(fconfig)
        for i,v in vars(args).items():
            if v is not None:
                config[i] = v
    else:
        config = vars(args)
    if 'lowervalues' not in config.keys():
        config['lowervalues'] = None
    if 'uppervalues' not in config.keys():
        config['uppervalues'] = None
    if 'desired_variance' not in config.keys():
        config['desired_variance'] = None

    if not os.path.isfile(config['modelfilepath']):
        print('Error: You should inform the correct module of objective '
              'function to model')
        sys.exit()

    if not os.path.isfile(config['parsecpyfilepath']):
        print('Error: You should inform the correct parsecpy measures file')
        sys.exit()


    parsec_exec = ParsecData(config['parsecpyfilepath'])
    y_measure = parsec_exec.speedups()

    n = y_measure.columns
    if config['problemsizes']:
        n_model = n.copy()
        for i in config['problemsizes']:
            if i not in n_model:
                print('Error: Measures not has especified problem sizes')
                sys.exit()
        y_measure = y_measure[config['problemsizes']]

    x = []
    y = []
    for i, row in y_measure.iterrows():
        for c, v in row.iteritems():
            x.append([i, int(c.split('_')[1])])
            y.append(v)
    input_name = c.split('_')[0]
    x = np.array(x)
    y = np.array(y)

    argsanneal = (config['overhead'], {'x': x, 'y': y, 'input_name': input_name})

    repetitions = range(config['repetitions'])
    err_min = 0
    computed_models = []
    best_model_idx = 0

    starttime = time.time()
    for i in repetitions:
        print('\nAlgorithm Execution: ', i+1)

        if config['lowervalues'] is None or config['uppervalues'] is None:
            initial_state = [tuple((random.normalvariate(0, 5) for _ in
                                    range(config['dimension'])))
                             for _ in range(config['annealers'])]
        else:
            initial_state = []
            for j in range(config['annealers']):
                t = tuple([li+(ui-li)*random.random() for li,ui
                           in zip(config['lowervalues'], config['uppervalues'])])
                initial_state.append(t)

        cann = CoupledAnnealer(
            initial_state,
            parsecpydatapath=config['parsecpyfilepath'],
            modelcodepath=config['modelfilepath'],
            n_annealers=config['annealers'],
            steps=config['steps'],
            update_interval=config['update_interval'],
            tgen_initial=config['tgen_initial'],
            tgen_upd_factor=config['tgen_upd_factor'],
            tacc_initial=config['tacc_initial'],
            alpha=config['alpha'],
            desired_variance=config['desired_variance'],
            pxmin=config['lowervalues'],
            pxmax=config['uppervalues'],
            threads=config['threads'],
            verbosity=config['verbosity'],
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
    if config['verbosity'] > 0:
        print('Best Parameters: \n', computed_models[best_model_idx].params)
    if config['verbosity'] > 1:
        print('\nMeasured Speedup: \n', y_measure)
        print('\nModeled Speedup: \n', computed_models[best_model_idx].y_model)

    print('\n***** Modelling Done! *****\n')

    if config['crossvalidation']:
        print('\n\n***** Starting cross validation! *****\n')
        starttime = time.time()

        validation_model = deepcopy(computed_models[best_model_idx])
        scores = validation_model.validate(kfolds=10)
        print('\n  Cross Validation (K-fold, K=10) Metrics: ')
        if config['verbosity'] > 2:
            print('\n   Times: ')
            for key, value in scores['times'].items():
                print('     %s: %.8f' % (key, value.mean()))
                print('     ', value)
        print('\n   Scores: ')
        for key, value in scores['scores'].items():
            if config['verbosity'] > 1:
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

    computed_models[best_model_idx].validation = deepcopy(validation_model.validation)
    fn = computed_models[best_model_idx].savedata(parsec_exec.config)
    print('Model data saved on filename: %s' % fn)


if __name__ == '__main__':
    main()

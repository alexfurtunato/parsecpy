#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel [-h] --config CONFIG -f PARSECPYFILEPATH
                             [-p PARTICLES] [-x MAXITERATIONS]
                             [-l LOWERVALUES] [-u UPPERVALUES]
                             [-n PROBLEMSIZES] [-o OVERHEAD] [-t THREADS]
                             [-r REPETITIONS] [-c CROSSVALIDATION]
                             [-v VERBOSITY]

    Script to run modelling algorithm to predict a parsec application output

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Filepath from Configuration file configurations
                            parameters
      -a ALGORITHM, --algorithm ['csa' or 'pso']
                            Optimization algorithm to use on modelling
                            process.
      -p PARSECPYFILEPATH, --parsecpyfilepath PARSECPYFILEPATH
                            Path from input data file from Parsec specificated
                            package.
      -f FREQUENCIES, --frequency FREQUENCIES
                            List of frequencies (KHz). Ex: 2000000, 2100000
      -n PROBLEMSIZES, --problemsizes PROBLEMSIZES
                            List of problem sizes to model used. Ex:
                            native_01,native_05,native_08
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to algorithm execution
      -m FRACTION, --measuresfraction FRACTION
                            Fraction of measures data to calculate the model
      -c CROSSVALIDATION, --crossvalidation CROSSVALIDATION
                            If run the cross validation of modelling
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose
    Example
        parsecpy_runmodel --config my_config.json -a pso
              -p /var/myparsecsim.dat -r 4 -v 3
"""

import os
import sys
import json
import time
import argparse
from copy import deepcopy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from parsecpy import ParsecData
from parsecpy import Swarm, CoupledAnnealer
from parsecpy import ParsecModel
from parsecpy import argsparselist, argsparseintlist, argsparsefraction
from parsecpy import data_detach


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run optimizer '
                                                 'modelling algorithm to '
                                                 'predict a parsec application '
                                                 'output')
    parser.add_argument('--config', required=True,
                        help='Filepath from Configuration file '
                             'configurations parameters')
    parser.add_argument('-a', '--algorithm', required=True,
                        choices=['csa', 'pso'],
                        help='Optimization algorithm to use on modelling'
                             'process.')
    parser.add_argument('-p', '--parsecpyfilepath',
                        help='Path from input data file from Parsec '
                             'specificated package.')
    parser.add_argument('-f', '--frequency', type=argsparseintlist,
                        help='List of frequencies (KHz). Ex: 2000000, 2100000')
    parser.add_argument('-n', '--problemsizes', type=argsparselist,
                        help='List of problem sizes to model used. '
                             'Ex: native_01,native_05,native_08')
    parser.add_argument('-o', '--overhead', type=bool,
                        help='If it consider the overhead')
    parser.add_argument('-t', '--threads', type=int,
                        help='Number of Threads')
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions to algorithm execution')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--crossvalidation', type=bool,
                       help='If run the cross validation of modelling')
    group.add_argument('-m', '--measuresfraction', type=argsparsefraction,
                       help='Fraction of measures data to calculate the model')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose')
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    # adjust list of arguments to avoid negative number values error
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg

    args = argsparsevalidation()

    print("\n***** Processing the Model *****")

    if args.config:
        if not os.path.isfile(args.config):
            print('Error: You should inform the correct config file path.')
            sys.exit()
        with open(args.config, 'r') as fconfig:
            config = json.load(fconfig)
        for i, v in vars(args).items():
            if v is not None:
                config[i] = v
    else:
        config = vars(args)

    if not os.path.isfile(config['modelfilepath']):
        print('Error: You should inform the correct module of '
              'objective function to model')
        sys.exit()

    if not os.path.isfile(config['parsecpyfilepath']):
        print('Error: You should inform the correct parsecpy measures file')
        sys.exit()

    lv = config['lowervalues']
    uv = config['uppervalues']

    parsec_exec = ParsecData(config['parsecpyfilepath'])
    y_measure = parsec_exec.speedups()
    input_sizes = []
    if 'size' in y_measure.dims:
        input_sizes = y_measure.attrs['input_sizes']
        input_ord = []
        if 'problemsizes' in config.keys():
                for i in config['problemsizes']:
                    if i not in input_sizes:
                        print('Error: Measures not has especified sizes')
                        sys.exit()
                    input_ord.append(input_sizes.index(i)+1)
                y_measure = y_measure.sel(size=sorted(input_ord))
                y_measure.attrs['input_sizes'] = sorted(config['problemsizes'])

    if 'frequency' in y_measure.dims:
        frequencies = y_measure.coords['frequency']
        if 'frequency' in config.keys():
                for i in config['frequency']:
                    if i not in frequencies:
                        print('Error: Measures not has especified frequencies')
                        sys.exit()
                y_measure = y_measure.sel(size=sorted(config['frequencies']))

    y_measure_detach = data_detach(y_measure)
    if 'measuresfraction' in config.keys():
        xy_train_test = train_test_split(y_measure_detach['x'],
                                         y_measure_detach['y'],
                                         test_size=1.0 - config['measuresfraction'])
        x_sample = xy_train_test[0]
        y_sample = xy_train_test[2]
    else:
        x_sample = y_measure_detach['x']
        y_sample = y_measure_detach['y']

    kwargsmodel = {'overhead': config['overhead']}

    repetitions = range(config['repetitions'])
    err_min = 0
    computed_models = []
    best_model_idx = 0

    starttime = time.time()
    for i in repetitions:
        print('\nAlgorithm Execution: ', i+1)

        if config['algorithm'] == 'pso':
            optm = Swarm(lv, uv, parsecpydatapath=config['parsecpyfilepath'],
                         modelcodepath=config['modelfilepath'],
                         size=config['particles'], w=config['w'],
                         c1=config['c1'], c2=config['c2'],
                         maxiter=config['maxiter'],
                         threads=config['threads'],
                         verbosity=config['verbosity'],
                         x_meas=x_sample, y_meas=y_sample,
                         kwargs=kwargsmodel)
        elif config['algorithm'] == 'csa':
            initial_state = np.array([np.random.uniform(size=config['dimension'])
                                      for _ in range(config['annealers'])])
            optm = CoupledAnnealer(initial_state,
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
                                   lowervalues=config['lowervalues'],
                                   uppervalues=config['uppervalues'],
                                   threads=config['threads'],
                                   verbosity=config['verbosity'],
                                   x_meas=x_sample,
                                   y_meas=y_sample,
                                   kwargs=kwargsmodel)
        else:
            print('Error: You should inform the correct algorithm to use')
            sys.exit()

        error, solution = optm.run()
        model = ParsecModel(bsol=solution,
                            berr=error,
                            ymeas=y_measure,
                            modelcodesource=optm.modelcodesource,
                            modelexecparams=optm.get_parameters())
        if 'measuresfraction' in config.keys():
            pred = model.predict(y_measure_detach['x'])
            model.error = mean_squared_error(y_measure_detach['y'], pred['y'])
            model.errorrel = 100 * (model.error / y_measure.values.mean())
            model.measuresfraction = config['measuresfraction']
        computed_models.append(model)
        if i == 0:
            err_min = model.error
            print('  Error: %.8f' % err_min)
        else:
            if model.error < err_min:
                best_model_idx = i
                print('  Error: %.8f -> %.8f ' % (err_min, model.error))
                err_min = model.error
        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))
        starttime = endtime

    print('\n\n***** Modelling Results! *****\n')
    print('Error: %.8f \nPercentual Error (Measured Mean): %.2f %%' %
          (computed_models[best_model_idx].error,
           computed_models[best_model_idx].errorrel))
    if config['verbosity'] > 0:
        print('Best Parameters: \n', computed_models[best_model_idx].sol)
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
        computed_models[best_model_idx].validation = scores
        print('\n***** Cross Validation Done! *****\n')
    print('\n\n***** ALL DONE! *****\n')
    fn = computed_models[best_model_idx].savedata(parsec_exec.config,
                                                  ' '.join(sys.argv))
    print('Model data saved on filename: %s' % fn)


if __name__ == '__main__':
    main()

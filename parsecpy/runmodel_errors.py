#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run optimzer modeller with differents number of measures.

    The aim is to visualize the minimal number of measures witch maintains the
    model error at acceptable limits

    usage: parsecpy_runmodel_errors [-h] -a ALGORITHM -m MODELFILENAME
                             [-r REPETITIONS] [-v VERBOSITY]

    Script to run pso modelling errors to find out a minimal number of measures
    to use on modelling

    optional arguments:
      -h, --help            show this help message and exit
      -m MODELFILENAME, --modelfilename MODELFILENAME
                            Absolute path from model filename with PSO Model
                            parameters executed previously.
      -f FOLDS, --folds FOLDS
                            Number of folds to use on split train and test
                            group of values
      -l LIMITS, --limits LIMITS
                            If include the surface limits points(4) on samples
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_errors -a pso -m /var/myparseccsamodel.dat
        --config /var/myconfig.json -r 10 -v 3
"""

import os
import sys
import time
import re
from datetime import datetime
import numpy as np
import json
import argparse
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from concurrent import futures
from parsecpy import Swarm, CoupledAnnealer, ParsecModel
from parsecpy import data_detach


def workers(args):
    config = args[0]
    y_measure = args[1]
    train = args[2]
    test = args[3]
    kwargsmodel = {'overhead': config['overhead']}

    if config['algorithm'] == 'pso':
        optm = Swarm(config['lowervalues'], config['uppervalues'],
                     parsecpydatapath=config['parsecpydatapath'],
                     modelcodepath=config['modelcodepath'],
                     size=config['size'], w=config['w'],
                     c1=config['c1'], c2=config['c2'],
                     maxiter=config['maxiter'],
                     threads=config['threads'],
                     verbosity=config['verbosity'],
                     x_meas=train['x'], y_meas=train['y'],
                     kwargs=kwargsmodel)
    elif config['algorithm'] == 'csa':
        initial_state = np.array([np.random.uniform(size=config['dimension'])
                                  for _ in range(config['annealers'])])
        optm = CoupledAnnealer(initial_state,
                               parsecpydatapath=config['parsecpydatapath'],
                               modelcodepath=config['modelcodepath'],
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
                               x_meas=train['x'],
                               y_meas=train['y'],
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
    pred = model.predict(test['x'])
    error = mean_squared_error(test['y'], pred['y'])
    train_list = {'x': list(train['x']), 'y': list(train['y'])}
    test_list = {'x': list(test['x']), 'y': list(test['y'])}
    return {'train': train_list, 'test': test_list,
            'error': error, 'sol': model.sol}


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run pso '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-m', '--modelfilepath',
                        help='Absolute path from model filename with '
                             'PSO Model parameters executed previously.')
    parser.add_argument('-f', '--folds', type=int,
                        help='Number of folds to use on split train and test '
                             'group of values')
    parser.add_argument('-l', '--limits', type=bool,
                        help='If include the surface limits points(4) on '
                             'samples', default=False)
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

    if not os.path.isfile(args.modelfilepath):
        print('Error: You should inform the correct module of objective '
              'function to model')
        sys.exit()

    parsec_model = ParsecModel(args.modelfilepath)
    tipo_modelo = re.search(r'config\d', parsec_model.modelcommand).group()
    tipo_modelo = tipo_modelo[-1]
    y_measure = parsec_model.y_measure
    config = parsec_model.modelexecparams
    if args.verbosity:
        config['verbosity'] = args.verbosity

    if 'size' in y_measure.dims:
        coord_2 = y_measure.coords['size']
    elif 'frequency' in y_measure.dims:
        coord_2 = y_measure.coords['frequency']

    y_measure_detach = data_detach(y_measure)

    if args.limits:
        # Separate max and min limits values of x and y on measures

        cores_limits = [y_measure.coords['cores'].values.min(),
                        y_measure.coords['cores'].values.max()]
        size_or_freq_limits = [coord_2.values.min(),
                               coord_2.values.max()]

        limits = []
        for sf in size_or_freq_limits:
            for c in cores_limits:
                limits.append([sf, c])

        limits_bool = np.isin(y_measure_detach['x'], limits)
        limits_bool = np.array([np.all(i) for i in limits_bool])

        x_limits = y_measure_detach['x'][limits_bool]
        x_without_limits = y_measure_detach['x'][~limits_bool]
        y_limits = y_measure_detach['y'][limits_bool]
        y_without_limits = y_measure_detach['y'][~limits_bool]

    computed_errors = []
    samples_n = 1
    for i in [len(y_measure.coords[i]) for i in y_measure.coords]:
        samples_n *= i
    if args.limits:
        train_size = max(len(parsec_model.sol), len(y_limits))
    else:
        train_size = len(parsec_model.sol)

    while True:

        print('\nSample size: ', train_size)

        samples_args = []
        sf = ShuffleSplit(n_splits=args.folds, train_size=train_size)
        if args.limits:
            for train_idx, test_idx in sf.split(x_without_limits):
                x_train = np.concatenate(
                    (x_limits, x_without_limits[train_idx]),
                    axis=0)
                y_train = np.concatenate(
                    (y_limits, y_without_limits[train_idx]))
                x_test = x_without_limits[test_idx]
                y_test = y_without_limits[test_idx]
                samples_args.append((config, y_measure,
                                     {'x': x_train,
                                      'y': y_train},
                                     {'x': x_test,
                                      'y': y_test}))
        else:
            for train_idx, test_idx in sf.split(y_measure_detach['x']):
                x_train = y_measure_detach['x'][train_idx]
                y_train = y_measure_detach['y'][train_idx]
                x_test = y_measure_detach['x'][test_idx]
                y_test = y_measure_detach['y'][test_idx]
                samples_args.append((config, y_measure,
                                     {'x': x_train,
                                      'y': y_train},
                                     {'x': x_test,
                                      'y': y_test}))

        print(' ** Args len = ', len(samples_args))
        starttime = time.time()

        with futures.ThreadPoolExecutor(max_workers=len(samples_args)) \
                as executor:
            results = executor.map(workers, samples_args)
            train = []
            test = []
            errors = []
            sols = []
            for i in results:
                train.append(i['train'])
                test.append(i['test'])
                errors.append(i['error'])
                sols.append(list(i['sol']))
            computed_errors.append({'train_size': train_size,
                                    'train': train,
                                    'test': test,
                                    'errors': errors,
                                    'sols': sols})

        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))
        if train_size >= int(samples_n/2):
            break
        train_size *= 2

    head = {'algorithm': config['algorithm'],
            'modeldatapath': args.modelfilepath,
            }
    print('\n\n***** Final Results *****\n')

    print(head)
    for i in computed_errors:
        print('Samples {0:2d}'.format(i['train_size']))
        print('  * Errors: {}'.format(i['errors']))
        print('  * Params: {}'.format(i['sols']))
    print('Samples {0:2d}'.format(samples_n))
    print('  * Errors: {0:.4f}'.format(parsec_model.error))
    print('  * Params: {}'.format(parsec_model.sol))

    filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pkgname = args.modelfilepath.split('_')[0]
    filename = '%s_%serrors_%s_%s.errordat' % (pkgname,
                                               config['algorithm'],
                                               tipo_modelo,
                                               filedate)
    with open(filename, 'w') as f:
        json.dump({'head': head, 'errors': computed_errors}, f,
                  ensure_ascii=False)
    print('Errors data saved on filename: %s' % filename)

    print('\n\n***** ALL DONE! *****\n')


if __name__ == '__main__':
    main()

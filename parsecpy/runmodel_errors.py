#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run optimzer modeller with differents number of measures.

    The aim is to visualize the minimal number of measures witch maintains the
    model error at acceptable limits

    usage: parsecpy_runmodel_errors [-h] -p PARSECPYDATAFILEPATH
                             [-v VERBOSITY]

    Script to run pso modelling errors to find out a minimal number of measures
    to use on modelling

    optional arguments:
      -h, --help            show this help message and exit
      -m MODELFILENAME, --modelfilename MODELFILENAME
                            Absolute path from model filename with Model
                            parameters executed previously.
      -f FOLDS, --folds FOLDS
                            Number of folds to use on split train and test
                            group of values
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_errors -m /var/myparseccsamodel.dat -v 3
"""

import os
import sys
import time
import re
from datetime import datetime
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from concurrent import futures
from parsecpy import Swarm, CoupledAnnealer, ParsecModel
from parsecpy import data_attach, data_detach


def workers(args):
    config = args[0]
    measure = args[1]
    train = args[2]
    test = args[3]

    if config['algorithm'] == 'svr':
        measure_svr = measure.copy()
        measure_svr.coords['frequency'] = measure_svr.coords['frequency']/1e6
        measure_svr_detach = data_detach(measure_svr)
        gs_svr = GridSearchCV(SVR(),
                              cv=config['crossvalidation-folds'],
                              param_grid={"C": config['c_grid'],
                                          "gamma": config['gamma_grid']})
        gs_svr.fit(train['x'], train['y'])
        y_predict = gs_svr.predict(test['x'])
        error = mean_squared_error(test['y'], y_predict)
        solution = gs_svr.best_params_
    else:
        kwargsmodel = {'overhead': config['overhead']}
        if config['algorithm'] == 'pso':
            optm = Swarm(config['lowervalues'], config['uppervalues'],
                         parsecpydatafilepath=config['parsecpydatafilepath'],
                         modelcodefilepath=config['modelcodefilepath'],
                         size=config['size'], w=config['w'],
                         c1=config['c1'], c2=config['c2'],
                         maxiter=config['maxiter'],
                         threads=config['threads'],
                         verbosity=config['verbosity'],
                         x_meas=train['x'], y_meas=train['y'],
                         kwargs=kwargsmodel)
        elif config['algorithm'] == 'csa':
            initial_state = np.array([np.random.uniform(size=config['dimension'])
                                      for _ in range(config['size'])])
            optm = CoupledAnnealer(initial_state,
                                   parsecpydatafilepath=config['parsecpydatafilepath'],
                                   modelcodefilepath=config['modelcodefilepath'],
                                   size=config['size'],
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
        solution = {'sol': list(solution)}
        model = ParsecModel(bsol=solution['sol'],
                            berr=error,
                            measure=measure,
                            modelcodesource=optm.modelcodesource,
                            modelexecparams=optm.get_parameters())
        pred = model.predict(test['x'])
        error = mean_squared_error(test['y'], pred['y'])
    train_list = {'x': train['x'].tolist(), 'y': train['y'].tolist()}
    test_list = {'x': test['x'].tolist(), 'y': test['y'].tolist()}
    return {'train': train_list, 'test': test_list,
            'error': error, 'params': solution}


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run pso '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-m', '--modelcodefilepath',
                        help='Absolute path from model filename with '
                             'PSO Model parameters executed previously.')
    parser.add_argument('-f', '--folds', type=int, default=10,
                        help='Number of folds to use on split train and test '
                             'group of values')
    parser.add_argument('-i', '--initnumber', type=int, default=4,
                        help='Number initial of measures number')
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose')
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    print("\n***** Processing the Models *****")

    args = argsparsevalidation()

    if not os.path.isfile(args.modelcodefilepath):
        print('Error: You should inform the correct parsec model data '
              'file path')
        sys.exit()

    parsec_model = ParsecModel(args.modelcodefilepath)
    config = parsec_model.modelexecparams
    if 'svr' in config['algorithm']:
        tipo_modelo = '5'
    else:
        tipo_modelo = re.search(r'config\d', parsec_model.modelcommand).group()
        tipo_modelo = tipo_modelo[-1]
    measure = parsec_model.measure
    if args.verbosity:
        config['verbosity'] = args.verbosity

    if 'size' in measure.dims:
        coord_2 = measure.coords['size']
    elif 'frequency' in measure.dims:
        coord_2 = measure.coords['frequency']

    measure_detach = data_detach(measure)

    computed_errors = []
    samples_n = 1
    for i in [len(measure.coords[i]) for i in measure.coords]:
        samples_n *= i
    train_size = args.initnumber

    while True:

        print('\nSample size: ', train_size)

        samples_args = []
        sf = ShuffleSplit(n_splits=args.folds, train_size=train_size,
                          test_size=(samples_n - train_size))
        for train_idx, test_idx in sf.split(measure_detach['x']):
            x_train = measure_detach['x'][train_idx]
            y_train = measure_detach['y'][train_idx]
            x_test = measure_detach['x'][test_idx]
            y_test = measure_detach['y'][test_idx]
            samples_args.append((config, measure,
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
            params = []
            for i in results:
                train.append(i['train'])
                test.append(i['test'])
                errors.append(i['error'])
                params.append(i['params'])
            computed_errors.append({'train_size': train_size,
                                    'train': train,
                                    'test': test,
                                    'errors': errors,
                                    'params': params})

        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))
        if train_size >= int(samples_n/2):
            break
        train_size *= 2

    head = {'algorithm': config['algorithm'],
            'modeldatapath': args.modelcodefilepath,
            }
    print('\n\n***** Final Results *****\n')

    print(head)
    for i in computed_errors:
        print('Samples {0:2d}'.format(i['train_size']))
        print('  * Train: {}'.format(i['train']))
        print('  * Test: {}'.format(i['test']))
        print('  * Errors: {}'.format(i['errors']))
        print('  * Params: {}'.format(i['params']))

    filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pkgname = args.modelcodefilepath.split('_')[0]
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

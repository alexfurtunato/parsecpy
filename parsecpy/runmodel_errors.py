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
      -a ALGORITHM, --algorithm ['csa' or 'pso']
                            Optimization algorithm to use on modelling
                            process.
      -m MODELFILENAME, --modelfilename MODELFILENAME
                            Absolute path from model filename with PSO Model
                            parameters executed previously.
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to each number of measures
                            algorithm execution
      -s SAMPLES, --samples SAMPLES
                            Number of samples to test
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_errors -a pso -m /var/myparseccsamodel.dat
        --config /var/myconfig.json -r 10 -v 3
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from concurrent import futures
from parsecpy import Swarm, CoupledAnnealer, ParsecModel
from parsecpy import data_detach


def workers(args):
    config = args[0]
    y_measure = args[1]
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
                     x_meas=args[2]['x'], y_meas=args[2]['y'],
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
                               x_meas=args[2]['x'],
                               y_meas=args[2]['y'],
                               kwargs=kwargsmodel)
    else:
        print('Error: You should inform the correct algorithm to use')
        sys.exit()
    error, solution = optm.run()
    y_measure_detach = data_detach(y_measure)
    model = ParsecModel(bsol=solution,
                        berr=error,
                        ymeas=y_measure,
                        modelcodesource=optm.modelcodesource,
                        modelexecparams=optm.get_parameters())
    pred = model.predict(y_measure_detach['x'])
    error = mean_squared_error(y_measure_detach['y'], pred['y'])
    return {'error': error, 'sol': model.sol}


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run pso '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-a', '--algorithm', required=True,
                        choices=['csa', 'pso'],
                        help='Optimization algorithm to use on modelling'
                             'process.')
    parser.add_argument('-m', '--modelfilepath',
                        help='Absolute path from model filename with '
                             'PSO Model parameters executed previously.')
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions to each number of measures '
                             'algorithm execution')
    parser.add_argument('-s', '--samples', type=int,
                        help='Number of samples to test')
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
    y_measure = parsec_model.y_measure
    config = parsec_model.modelexecparams
    if args.verbosity:
        config['verbosity'] = args.verbosity

    input_sizes = []
    if 'size' in y_measure.dims:
        coord_2 = y_measure.coords['size']
    elif 'frequency' in y_measure.dims:
        coord_2 = y_measure.coords['frequency']

    y_measure_detach = data_detach(y_measure)

    # Separate max and min limits values of x and y on measures

    # cores_limits = [y_measure.coords['cores'].values.min(),
    #                 y_measure.coords['cores'].values.max()]
    # size_or_freq_limits = [coord_2.values.min(),
    #                        coord_2.values.max()]

    # limits = []
    # for sf in size_or_freq_limits:
    #     for c in cores_limits:
    #         limits.append([sf, c])

    # limits_bool = np.isin(y_measure_detach['x'], limits)
    # limits_bool = np.array([np.all(i) for i in limits_bool])

    # x_limits = y_measure_detach['x'][limits_bool]
    # x_without_limits = y_measure_detach['x'][~limits_bool]
    # y_limits = y_measure_detach['y'][limits_bool]
    # y_without_limits = y_measure_detach['y'][~limits_bool]

    computed_errors = []
    repetitions = range(args.repetitions)
    samples_n = args.samples

    for k in range(1,samples_n):

        print('\nAlgorithm Execution: ', k)

        samples_args = []
        for i in repetitions:
            # xy_train_test = train_test_split(x_without_limits,
            #                                  y_without_limits,
            #                                  test_size=(samples_n-(k+1))/samples_n)
            # print(' ** ', i, ' - samples lens: x=', len(xy_train_test[0]), ', y=', len(xy_train_test[2]))
            # x_sample = np.concatenate((x_limits, xy_train_test[0]), axis=0)
            # y_sample = np.concatenate((y_limits, xy_train_test[2]))
            test_size = 1 - (k/samples_n)
            xy_train_test = train_test_split(y_measure_detach['x'],
                                             y_measure_detach['y'],
                                             test_size=test_size)
            print(' ** ', i, ' - samples lens: x=', len(xy_train_test[0]), ', y=', len(xy_train_test[2]))
            x_sample = xy_train_test[0]
            y_sample = xy_train_test[2]
            samples_args.append((config, y_measure,
                                 {'x': x_sample,
                                  'y': y_sample}))
        print(' ** Args len = ', len(samples_args))
        starttime = time.time()

        with futures.ThreadPoolExecutor(max_workers=args.repetitions) \
                as executor:
                results = executor.map(workers, samples_args)
                errors = []
                sols = []
                for i in results:
                    errors.append(i['error'])
                    sols.append(list(i['sol']))
                computed_errors.append({'k': k+1,
                                        'errors': errors,
                                        'sols': sols})

        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))

    print('\n\n***** Final Results *****\n')

    for i in computed_errors:
        print('Iteration {0:2d}'.format(i['k']))
        print('  * Errors: {}'.format(i['errors']))
        print('  * Params: {}'.format(i['sols']))
    print('Iteration {0:2d}'.format(10))
    print('  * Errors: {0:.4f}'.format(parsec_model.error))
    print('  * Params: {}'.format(parsec_model.sol))

    filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pkgname = args.modelfilepath.split('_')[0]
    filename = '%s_%serrors_%s.errordat' % (pkgname, args.algorithm, filedate)
    with open(filename, 'w') as f:
        json.dump(computed_errors, f, ensure_ascii=False)
    print('Errors data saved on filename: %s' % filename)

    print('\n\n***** ALL DONE! *****\n')

if __name__ == '__main__':
    main()
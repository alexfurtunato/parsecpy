#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run csa models with differents number of measures.

    The aim is to visualize the minimal number of measures witch maintains the
    model error at acceptable limits

    usage: parsecpy_runmodel_errorscsa [-h] -m MODELFILENAME
                             [-r REPETITIONS] [-v VERBOSITY]

    Script to run csa modelling errors to find out a minimal number of measures
    to use on modelling

    optional arguments:
      -h, --help            show this help message and exit
      -m MODELFILENAME, --modelfilename MODELFILENAME
                            Absolute path from model filename with CSA Model
                            parameters executed previously.
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to each number of measures
                            algorithm execution
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_errorscsa -m /var/myparseccsamodel.dat
        --config /var/myconfig.json -r 10 -v 3
"""

import os
import sys
import time
from datetime import datetime
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent import futures
from parsecpy import CoupledAnnealer, ModelCoupledAnnealer
from parsecpy import data_detach


def workers(args):
    config = args[0]
    initial_state = np.array([np.random.uniform(size=config['dimension'])
                              for _ in range(config['m'])])
    argsanneal = (config['oh'], args[1])

    cann = CoupledAnnealer(
        initial_state,
        parsecpydatapath=config['parsecpydatapath'],
        modelcodepath=config['modelcodepath'],
        n_annealers=config['m'],
        steps=config['steps'],
        update_interval=config['update_interval'],
        tgen_initial=config['tgen'],
        tgen_upd_factor=config['tgen_upd_factor'],
        tacc_initial=config['tacc'],
        alpha=config['alpha'],
        desired_variance=config['desired_variance'],
        pxmin=config['pxmin'],
        pxmax=config['pxmax'],
        threads=config['threads'],
        verbosity=config['verbosity'],
        args=argsanneal
    )
    model = cann.run()
    return model.error


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run csa '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-m', '--modelfilepath',
                        help='Absolute path from model filename with '
                             'CSA Model parameters executed previously.')
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions to each number of measures '
                             'algorithm execution')
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

    parsec_model = ModelCoupledAnnealer(args.modelfilepath)
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
    repetitions = range(args.repetitions)
    samples_args = []
    samples_n = 10

    for k in range(samples_n-1):

        print('\nAlgorithm Execution: ', k+1)

        for _ in repetitions:
            xy_train_test = train_test_split(x_without_limits,
                                             y_without_limits,
                                             test_size=(samples_n-(k+1))/samples_n)
            x_sample = np.concatenate((x_limits, xy_train_test[0]), axis=0)
            y_sample = np.concatenate((y_limits, xy_train_test[2]))
            samples_args.append((config,
                                 {'x': x_sample,
                                  'y': y_sample,
                                  'dims': y_measure.dims,
                                  'input_sizes': input_sizes}))
        x_sample = np.concatenate((x_limits, x_without_limits), axis=0)
        y_sample = np.concatenate((y_limits, y_without_limits))
        samples_args.append((config,
                                 {'x': x_sample,
                                  'y': y_sample,
                                  'dims': y_measure.dims,
                                  'input_sizes': input_sizes}))
        starttime = time.time()

        with futures.ThreadPoolExecutor(max_workers=args.repetitions) \
                as executor:
                results = executor.map(workers, samples_args)
                res = [i for i in results]
                computed_errors.append({'k': k+1, 'errors': res})

        endtime = time.time()
        print('  Execution time = %.2f seconds' % (endtime - starttime))

    print('\n\n***** Final Results *****\n')

    for i in computed_errors:
        print(i)

    filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    pkgname = args.modelfilepath.split('_')[0]
    filename = '%s_csaerrors_%s.dat' % (pkgname, filedate)
    with open(filename, 'w') as f:
        json.dump(computed_errors, f, ensure_ascii=False)
    print('Errors data saved on filename: %s' % filename)

    print('\n\n***** ALL DONE! *****\n')

if __name__ == '__main__':
    main()
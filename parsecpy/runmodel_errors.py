#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run optimzer modeller with differents number of measures.

    The aim is to visualize the minimal number of measures witch maintains the
    model error at acceptable limits

    usage: parsecpy_runmodel_errors [-h] -p PARSECPYDATAFILEPATH
                             [-v VERBOSITY]

    Script to run multiples runs with differents number of train points
    to modelling the application

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Filepath from Configuration file configurations
                            parameters
      -p PARSECPYDATAFILEPATH, --parsecpydatafilepath PARSECPYDATAFILEPATH
                            Path from input data file from Parsec specificated
                            package.
      -i INITNUMBER, --initnumber INITNUMBER
                            Number initial of measures number
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example
        parsecpy_runmodel_errors --config meuconfig.json
                                 -p /var/myparsecdata.dat -v 3
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import json
import argparse
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from concurrent import futures
from parsecpy import Swarm, CoupledAnnealer, ParsecData, ParsecModel
from parsecpy import data_detach


def workers(args):
    name = args[0]
    config = args[1]
    measure = args[2]
    measure_detach = data_detach(measure)
    train_idx = args[3]
    test_idx = args[4]
    x_train = measure_detach['x'][train_idx]
    y_train = measure_detach['y'][train_idx]
    x_test = measure_detach['x'][test_idx]
    y_test = measure_detach['y'][test_idx]

    if config['algorithm'] == 'svr':
        measure_svr = measure.copy()
        measure_svr.coords['frequency'] = measure_svr.coords['frequency']/1e6
        measure_svr_detach = data_detach(measure_svr)
        x_train = measure_svr_detach['x'][train_idx]
        y_train = measure_svr_detach['y'][train_idx]
        x_test = measure_svr_detach['x'][test_idx]
        y_test = measure_svr_detach['y'][test_idx]
        gs_svr = GridSearchCV(SVR(),
                              cv=config['crossvalidation-folds'],
                              param_grid={"C": config['c_grid'],
                                          "gamma": config['gamma_grid']})
        gs_svr.fit(x_train, y_train)
        y_predict = gs_svr.predict(x_test)
        error = mean_squared_error(y_test, y_predict)
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
                         x_meas=x_train, y_meas=y_train,
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
                                   x_meas=x_train,
                                   y_meas=y_train,
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
        pred = model.predict(x_test)
        error = mean_squared_error(y_test, pred['y'])
    train_list = {'x': x_train.tolist(), 'y': y_train.tolist()}
    test_list = {'x': x_test.tolist(), 'y': y_test.tolist()}
    return {'name': name,
            'train': train_list, 'test': test_list,
            'dims': measure_detach['dims'],
            'error': error, 'params': solution}


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run multiples '
                                                 'runs with differents number '
                                                 'of train points to modelling '
                                                 'the application')
    parser.add_argument('--config', required=True,
                        help='Filepath from Configuration file '
                             'configurations parameters')
    parser.add_argument('-p', '--parsecpydatafilepath',
                        help='Path from input data file from Parsec '
                             'specificated package.')
    parser.add_argument('-i', '--initnumber', type=int,
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

    if not os.path.isfile(config['parsecpydatafilepath']):
        print('Error: You should inform the correct parsecpy measures file')
        sys.exit()

    parsec_exec = ParsecData(config['parsecpydatafilepath'])
    measure = parsec_exec.speedups()
    measure_detach = data_detach(measure)

    # parsec_model = ParsecData(args.modelcodefilepath)
    # config = parsec_model.modelexecparams
    # if 'svr' in config['algorithm']:
    #     tipo_modelo = '5'
    # else:
    #     tipo_modelo = re.search(r'config\d', parsec_model.modelcommand).group()
    #     tipo_modelo = tipo_modelo[-1]
    # measure = parsec_model.measure
    # if args.verbosity:
    #     config['verbosity'] = args.verbosity
    #
    #
    # computed_errors = []

    samples_n = 1
    for i in [len(measure.coords[i]) for i in measure.coords]:
        samples_n *= i
    train_size = config['initnumber']

    model_results = {}
    for m in config['models']:
        with open(m["conf_file"]) as f:
            model_config = json.load(f)
        model_config["verbosity"] = config["verbosity"]

        model_results[m["name"]] = {
            "algorithm": model_config['algorithm'],
            "configfilepath": m["conf_file"],
            "config": model_config,
            "data": {}
        }

    while True:

        print('\nSample size: ', train_size)

        sf = ShuffleSplit(n_splits=config["numberofsplits"],
                          train_size=train_size,
                          test_size=(samples_n - train_size))
        splits = []
        split_n = 1
        for train_idx, test_idx in sf.split(measure_detach['x']):

            print(' ** # split = ', split_n)

            samples_args = [(m[0], m[1]['config'], measure,
                             train_idx, test_idx)
                            for m in model_results.items()]

            starttime = time.time()

            with futures.ProcessPoolExecutor(max_workers=len(samples_args)) \
                    as executor:
                results = executor.map(workers, samples_args)
                for i in results:
                    if train_size not in model_results[i["name"]]["data"].keys():
                        model_results[i["name"]]["data"][train_size] = {
                            'train': [],
                            'test': [],
                            'error': [],
                            'params': []
                        }
                    model_results[i["name"]]["data"][train_size]["train"].append(i['train'])
                    model_results[i["name"]]["data"][train_size]["test"].append(i['test'])
                    model_results[i["name"]]["data"][train_size]["error"].append(i['error'])
                    model_results[i["name"]]["data"][train_size]["params"].append(i['params'])

            endtime = time.time()
            print('  Execution time = %.2f seconds' % (endtime - starttime))

        if train_size >= int(samples_n/2):
            break
        train_size *= 2

    print('\n\n***** Final Results *****\n')

    for name, m in model_results.items():
        print("Model Name: {}".format(name))
        for ts, d in m["data"].items():
            print(' Train Size: {0:2d}'.format(ts))
            print('  * Median of Errors: {}'.format(np.median(d['error'])))

        filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        pkgname = os.path.basename(config['parsecpydatafilepath']).split('_')[0]
        filename = '%s_%s_errors_%s.errordat' % (pkgname,
                                                name,
                                                filedate)
        with open(filename, 'w') as f:
            json.dump({"parsecpydatafilepath": config['parsecpydatafilepath'],
                       "measure_dims": measure_detach['dims'],
                       "model_config": m["config"],
                       'data': m["data"]},
                      f,
                      ensure_ascii=False)
        print('Errors data saved on filename: %s' % filename)

    print('\n\n***** ALL DONE! *****\n')


if __name__ == '__main__':
    main()

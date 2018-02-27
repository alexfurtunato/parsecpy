#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a model cross validation based on previous saved run modelling.

    usage: parsecpy_runmodel_validation_pso [-h] -f MODELDATAFILENAME

    Script to run swarm modelling to predict aparsec application output

    optional arguments:
      -h, --help            show this help message and exit
      -f MODELDATAFILENAME, --modeldatafilename MODELDATAFILENAME
                            Input filename with data from model data run.

    Example
        parsecpy_runmodel_pso -f /var/myparsecsim.dat
"""

import os
import sys
import time
import argparse
import numpy as np
from parsecpy import ModelSwarm
from parsecpy import Swarm
from sklearn.base import BaseEstimator
from sklearn import cross_validation as cval

class PSOModel(BaseEstimator):

    def __init__(self):
        self.model = Swarm()

    def fit(self, X, y):
        self.model.train(X,y)

    def predict(self, X):
        return X

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run a cross '
                                                 'validation of a saved model')
    parser.add_argument('-f','--modeldatafilename', required=True,
                        help='Input filename with data from model data run.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    print("Processing the Cross Validation of Model")

    args = argsparsevalidation()

    if os.path.isfile(args.modeldatafilename):
        modelpath = args.modeldatafilename
    else:
        print('Error: You should inform the correct model data file.')
        sys.exit()

    parsec_data = ModelSwarm(args.parsecpyfilename)

    x = []
    y = []
    for i, row in parsec_data.y_measure.iterrows():
        for c, v in row.iteritems():
            x.append((i,int(c.split('_')[1])))
            y.append(v)
    input_name = c.split('_')[0]

    argsswarm = (args.overhead, {'x': x, 'y': y, 'input_name': input_name})


    starttime = time.time()


    endtime = time.time()

    print('Execution time = %s seconds' % (endtime - starttime))

    print('\n\n***** Done! *****\n')
    print('Error: %.8f \nPercentual Error (Measured Mean): %.8f %%' %
          (model.error,
           model.errorrel))
    print('Best Parameters: \n',model.params)
    print('\nMeasured Speedup: \n',y_measure)
    print('\nModeled Speedup: \n',model.y_model)

    fn = model.savedata(parsec_exec.config)
    print('Model data saved on filename: %s' % fn)

if __name__ == '__main__':
    main()
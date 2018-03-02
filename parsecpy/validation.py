# -*- coding: utf-8 -*-
"""
    Module with Classes to validating the model of an application

"""

from parsecpy import Swarm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SwarmEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, modeldata = None, verbosity=0):
        self.modeldata = modeldata
        self.verbosity = verbosity

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y)
        if self.verbosity > 2:
            print('\nFit: X lenght = ', X.shape,' y lenght = ',y.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        p = self.modeldata.modelexecparams.copy()
        args = (p['args'][0], {'x': X, 'y': y, 'input_name': p['args'][1]['input_name']})
        S = Swarm(p['pxmin'], p['pxmax'], args=args, threads=p['threads'],
                  size=p['size'], w=p['w'], c1=p['c1'], c2=p['c2'],
                  maxiter =p['maxiter'], modelcodesource=self.modeldata.modelcodesource, verbosity=self.verbosity)
        self.modeldata = S.run()
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        y = self.modeldata.predict(X)[1]
        if self.verbosity > 2:
            print('\nPredict: X lenght = ', X.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        return y


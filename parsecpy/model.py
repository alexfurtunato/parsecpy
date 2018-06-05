from datetime import datetime
import os
import numpy as np
import json
from copy import deepcopy
import xarray as xr

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import ParsecData, Swarm, CoupledAnnealer
from parsecpy import data_attach, data_detach

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    support3d = False


class ParsecModel:
    """
    Class of results of modelling using an optimization algorithm

        Atrributes
            sol - solution of optimizer (model parameters)
            error - model error or output of objective function
            y_measure - measured speedups xarray
            y_model - modeled speedups xarray

        Methods
            loadata()
            savedata()
            times()
            speedups()
            plot2D()
            plot3D

    """

    def __init__(self,
                 modeldatapath=None,
                 bsol=None,
                 berr=None,
                 ymeas=None,
                 modelcodepath=None,
                 modelcodesource=None,
                 modelexecparams=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param modeldatapath: path of the model data previously saved.
        :param bsol: best solution of optmizer
        :param berr: best error of optimizer
        :param ymeas: output speedup model calculated by model parameters
        :param modelcodepath: path of the python module with model coded.
        :param modelcodesource: python module source file content.
        :param modelexecparams: Model execution used parameters.
        """

        if modeldatapath:
            self.loaddata(modeldatapath)
        else:
            self.y_measure = ymeas
            self.modelexecparams = modelexecparams
            self.modelcodesource = None
            self.validation = None
            if modelcodepath is not None:
                f = open(modelcodepath)
                self.modelcodesource = f.read()
            if modelcodesource is not None:
                self.modelcodesource = modelcodesource
            if bsol is None:
                self.sol = None
                self.error = None
            else:
                self.sol = bsol
                self.error = berr
                self.errorrel = 100*(self.error/self.y_measure.values.mean())
            y_meas_detach = data_detach(self.y_measure)
            self.y_model = data_attach(self.predict(y_meas_detach['x']),
                                       y_meas_detach['dims'])

    @staticmethod
    def loadcode(codetext, modulename):
        """
        Load a python module stored on a string.

        :param codetext: string with model alghorithm.
        :param modulename: name of module to load with python code.
        :return: return module object with model alghorithm
        """
        import types

        module = types.ModuleType(modulename)
        exec(codetext, module.__dict__)
        return module

    def get_parsecdata(self):
        """
        Return a ParsecData object with measures

        :return: ParsecData object
        """
        if os.path.isfile(self.modelexecparams['parsecpydatapath']):
            pd = ParsecData(self.modelexecparams['parsecpydatapath'])
            return pd
        else:
            print('Parsecpy datarun file %s was not found')
            return None

    def predict(self, x):
        """
        Predict the speedup using the input values
        and based on mathematical model.

        :param x: array with number of cores and input size
                     or array of array.
        :return: return a array with speedup values.
        """

        if self.sol is None:
            print('Error: You should run or load the model data '
                  'before make predictions!')
            return None
        if not isinstance(x, np.ndarray):
            print('Error: The inputs should be a array (number of cores, '
                  'input size) or a array of array')
            return None
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        psomodel = self.loadcode(self.modelcodesource, 'psomodel')
        pred = psomodel.model(self.sol, x, self.modelexecparams['overhead'])
        return pred

    def validate(self, kfolds=3, scoring=None):
        """
        Validate the model.

        :param kfolds: number of folds to divide for cross-validate run.
        :param scoring: dictionary with defined scores to calculate
        :return: return dictionary with calculated scores.
        """

        kf = KFold(n_splits=kfolds, shuffle=True)

        if scoring is None:
            scoring = {
                'description': {
                    'test_neg_mse_error': 'Mean Squared Error',
                    'test_neg_mae_error': 'Mean Absolute Error',
                    'test_see_error': 'Standard Error of Estimate'
                },
                'type': {
                    'neg_mse_error': 'neg_mean_squared_error',
                    'neg_mae_error': 'neg_mean_absolute_error',
                    'see_error': make_scorer(ModelEstimator.see_score,
                                             parameters_number=len(
                                                 self.sol))
                }
            }
        y_measure_detach = data_detach(self.y_measure)
        scores = cross_validate(ModelEstimator(self,
                                verbosity=self.modelexecparams['verbosity']),
                                y_measure_detach['x'],
                                y_measure_detach['y'],
                                cv=kf, scoring=scoring['type'],
                                return_train_score=False,
                                verbose=self.modelexecparams['verbosity'])
        self.validation = {
            'times': {
                'fit_time': scores['fit_time'],
                'score_time': scores['score_time']
            },
            'scores': {}
        }
        for key, value in scores.items():
            if key not in ['fit_time', 'score_time']:
                if '_neg_' in key:
                    value = np.negative(value)
                self.validation['scores'][key] = {
                    'value': value,
                    'description': scoring['description'][key]
                }
        return self.validation

    def savedata(self, parsecconfig, modelcommand):
        """
        Write to a file the model information stored on object class

        :param parsecconfig: Configuration dictionary from parsecpy runprocess
        :param modelcommand: string with model executed command
        :return: saved file name
        """

        filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        pkgname = os.path.basename(self.modelexecparams['parsecpydatapath'])
        pkgname = pkgname.split('_')[0]
        filename = '%s_%smodel_datafile_%s.dat' % (pkgname,
                                                   self.modelexecparams['algorithm'],
                                                   filedate)
        with open(filename, 'w') as f:
            datatosave = {'config': {}, 'data': {}}
            if 'pkg' in parsecconfig:
                datatosave['config']['pkg'] = parsecconfig['pkg']
            if 'command' in parsecconfig:
                datatosave['config']['command'] = parsecconfig['command']
            datatosave['config']['modelcommand'] = modelcommand
            if 'hostname' in parsecconfig:
                datatosave['config']['hostname'] = parsecconfig['hostname']
            datatosave['config']['savedate'] = filedate
            datatosave['config']['modelcodesource'] = self.modelcodesource
            mep = deepcopy(self.modelexecparams)
            mep['lowervalues'] = str(mep['lowervalues'])
            mep['uppervalues'] = str(mep['uppervalues'])
            datatosave['config']['modelexecparams'] = mep
            datatosave['data']['params'] = str(list(self.sol))
            datatosave['data']['error'] = self.error
            datatosave['data']['errorrel'] = self.errorrel
            datatosave['data']['parsecdata'] = self.y_measure.to_dict()
            datatosave['data']['speedupmodel'] = self.y_model.to_dict()
            if hasattr(self, 'measuresfraction'):
                datatosave['config']['measuresfraction'] = self.measuresfraction
            if hasattr(self, 'validation'):
                if self.validation:
                    val = deepcopy(self.validation)
                    for key, value in val['scores'].items():
                        val['scores'][key]['type'] = str(value['value'].dtype)
                        val['scores'][key]['value'] = json.dumps(
                            value['value'].tolist())
                    for key, value in val['times'].items():
                        valtemp = {'type': '', 'value': ''}
                        valtemp['type'] = str(value.dtype)
                        valtemp['value'] = json.dumps(value.tolist())
                        val['times'][key] = valtemp.copy()
                    datatosave['data']['validation'] = val
            json.dump(datatosave, f, ensure_ascii=False)
        return filename

    def loaddata(self, filename):
        """
        Read a file previously saved with method savedata() and initialize
        the object class dictionaries.

        :param filename: File name with data dictionary of execution times.
        """

        if os.path.isfile(filename):
            with open(filename) as f:
                loaddict = json.load(f)
                datadict = loaddict['data']
                configdict = loaddict['config']
            if 'pkg' in configdict.keys():
                self.pkg = configdict['pkg']
            if 'command' in configdict.keys():
                self.command = configdict['command']
            if 'modelcommand' in configdict.keys():
                self.modelcommand = configdict['modelcommand']
            if 'hostname' in configdict.keys():
                self.hostname = configdict['hostname']
            if 'modelcodesource' in configdict.keys():
                self.modelcodesource = configdict['modelcodesource']
            if 'modelexecparams' in configdict.keys():
                mep = deepcopy(configdict['modelexecparams'])
                self.modelexecparams = deepcopy(mep)
                self.modelexecparams['lowervalues'] = json.loads(mep['lowervalues'])
                self.modelexecparams['uppervalues'] = json.loads(mep['uppervalues'])
            if 'measuresfraction' in configdict.keys():
                self.measuresfraction = configdict['measuresfraction']
            if 'validation' in datadict.keys():
                val = deepcopy(datadict['validation'])
                for key, value in val['scores'].items():
                    val['scores'][key]['value'] = \
                        np.array(json.loads(value['value']),
                                 dtype=value['type'])
                    val['scores'][key].pop('type')
                for key, value in val['times'].items():
                    val['times'][key] = \
                        np.array(json.loads(value['value']),
                                 dtype=value['type'])
                self.validation = val
            if 'params' in datadict.keys():
                self.sol = json.loads(datadict['params'])
            if 'error' in datadict.keys():
                self.error = datadict['error']
            if 'errorrel' in datadict.keys():
                self.errorrel = datadict['errorrel']
            if 'parsecdata' in datadict.keys():
                self.y_measure = xr.DataArray.from_dict(datadict['parsecdata'])
            if 'speedupmodel' in datadict.keys():
                self.y_model = xr.DataArray.from_dict(datadict['speedupmodel'])
            if 'savedate' in configdict.keys():
                self.savedate = configdict['savedate']
        else:
            print('Error: File not found')
            return
        return

    def plot3D(self, title='Speedup Model', greycolor=False,
               showmeasures=False, alpha=1.0, filename=''):
        """
        Plot the 3D (Speedup x cores x input size) surface.

        :param title: Plot Title.
        :param greycolor: If set color of graph to grey colormap.
        :param filename: File name to save figure (eps format).
        :return:
        """

        if not support3d:
            print('Warning: No 3D plot support. Please install '
                  'matplotlib with Axes3D toolkit')
            return
        data = self.y_model
        if not data.size == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            if 'size' in data.dims:
                xc = data.coords['size'].values
                xc_label = 'Input Size'
            elif 'frequency' in data.dims:
                xc = 1000*data.coords['frequency'].values
                xc_label = 'Frequency'
            yc = data.coords['cores'].values
            X, Y = np.meshgrid(yc, xc)
            Z = data.values
            zmin = Z.min()
            zmax = Z.max()
            appname = self.pkg
            plt.title('%s\n%s' % (appname.capitalize() or None, title))
            if greycolor:
                colormap = cm.Greys
            else:
                colormap = cm.coolwarm
            surf1 = ax.plot_surface(Y, X, Z, label='Model', cmap=colormap, linewidth=0.5,
                            edgecolor='k', linestyle='-', alpha=alpha)
            surf1._edgecolors2d = surf1._edgecolors3d
            surf1._facecolors2d = surf1._facecolors3d
            ax.set_xlabel(xc_label)
            if xc_label is 'Frequency':
                ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='Hz'))
            ax.set_ylabel('Number of Cores')
            ax.set_zlabel('Model Speedup')
            ax.set_zlim(zmin, 1.10 * zmax)
            if showmeasures:
                data_m = self.y_measure
                ax = fig.gca(projection='3d')
                if 'size' in data_m.dims:
                    xc = data_m.coords['size'].values
                elif 'frequency' in data_m.dims:
                    xc = 1000*data_m.coords['frequency'].values
                yc = data_m.coords['cores'].values
                X, Y = np.meshgrid(yc, xc)
                Z = data_m.values
                surf2 = ax.plot_wireframe(Y, X, Z, linewidth=0.5,
                                          edgecolor='k', label='Measures')
                x = np.repeat(xc, len(yc))
                y = np.resize(yc, len(xc)*len(yc))
                z = Z.flatten()
                ax.scatter(x, y, z, c='k', s=6)
                ax.set_zlim(min(zmin, Z.min()), 1.10 * max(zmax, Z.max()))
            ax.legend()
            if filename:
                plt.savefig(filename, format='eps', dpi=1000)
            plt.show()


class ModelEstimator(BaseEstimator, RegressorMixin):
    """
    Class of estimator to use on Cross Validation process

        Atrributes
            modeldata - ParsecModel object with modelling results
            verbosity - verbosity level of run

        Methods
            fit()
            predict()
            see_score()

    """

    def __init__(self, model=None, verbosity=0):
        self.model = model
        self.verbosity = verbosity

    def fit(self, X, y, **kwargs):
        """
        method to train of model with train splited data.

        :param X: measured inputs array
        :param y: measured outputs array
        :return: return ModelEstimator object.
        """

        X, y = check_X_y(X, y)
        if self.verbosity > 2:
            print('\nFit: X lenght = ', X.shape, ' y lenght = ', y.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        p = deepcopy(self.model.modelexecparams)
        kwargsmodel = {'overhead': p['overhead']}

        y_measure = data_detach(self.model.y_measure)

        if p['algorithm'] == 'pso':
            optm = Swarm(p['lowervalues'], p['uppervalues'],
                         parsecpydatapath=p['parsecpydatapath'],
                         modelcodesource=self.model.modelcodesource,
                         size=p['size'], w=p['w'], c1=p['c1'], c2=p['c2'],
                         maxiter=p['maxiter'], threads=p['threads'],
                         x_meas=y_measure['x'], y_meas=y_measure['y'],
                         verbosity=self.verbosity,
                         kwargs=kwargsmodel)
        elif p['algorithm'] == 'csa':
            initial_state = np.array([np.random.uniform(size=p['dimension'])
                                      for _ in range(p['annealers'])])
            optm = CoupledAnnealer(initial_state,
                                   parsecpydatapath=p['parsecpydatapath'],
                                   modelcodesource=self.model.modelcodesource,
                                   n_annealers=p['annealers'],
                                   steps=p['steps'],
                                   update_interval=p['update_interval'],
                                   tgen_initial=p['tgen_initial'],
                                   tgen_upd_factor=p['tgen_upd_factor'],
                                   tacc_initial=p['tacc_initial'],
                                   alpha=p['alpha'],
                                   desired_variance=p['desired_variance'],
                                   lowervalues=p['lowervalues'],
                                   uppervalues=p['uppervalues'],
                                   threads=p['threads'],
                                   verbosity=self.verbosity,
                                   kwargs=kwargsmodel)
        else:
            print('Error: You should inform the correct algorithm to use')
            return

        error, solution = optm.run()
        self.model = ParsecModel(bsol=solution,
                                 berr=error,
                                 ymeas=self.model.y_measure,
                                 modelcodesource=optm.modelcodesource,
                                 modelexecparams=optm.get_parameters())
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        method to test of model with test divided data.

        :param X: inputs array
        :return: return model outputs array.
        """

        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        y = self.model.predict(X)['y']
        if self.verbosity > 2:
            print('\nPredict: X lenght = ', X.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        return y

    @staticmethod
    def see_score(y, y_pred, **kwargs):
        """
        method to caclculate the "Standard Error of Estimator" score to use
        on cross validation process.

        :param y: measured outputs array
        :param y_pred: model outputs array
        :return: return caclulated score.
        """

        mse = mean_squared_error(y, y_pred)
        n = len(y)
        p = kwargs['parameters_number']
        see = np.sqrt(n * mse / (n - p))
        return see

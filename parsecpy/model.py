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
            measure - measured speedups xarray
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
                 berr_rel=None,
                 measure=None,
                 y_model=None,
                 modelcodefilepath=None,
                 modelcodesource=None,
                 modelexecparams=None,
                 modelresultsfolder=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param modeldatapath: path of the model data previously saved.
        :param bsol: best solution of optmizer
        :param berr: best error of optimizer
        :param measure: output speedup model calculated by model parameters
        :param modelcodefilepath: path of the python module with model coded.
        :param modelcodesource: python module source file content.
        :param modelexecparams: Model execution used parameters.
        """

        if modeldatapath:
            self.loaddata(modeldatapath)
        else:
            self.measure = measure
            self.modelexecparams = modelexecparams
            self.modelresultsfolder = modelresultsfolder
            self.modelcodesource = None
            self.validation = None
            if modelcodefilepath is not None:
                f = open(modelcodefilepath)
                self.modelcodesource = f.read()
            if modelcodesource is not None:
                self.modelcodesource = modelcodesource
            self.sol = bsol
            self.error = berr
            if self.error is not None:
                self.errorrel = 100*(self.error/self.measure.values.mean())
            if berr_rel is not None:
                self.errorrel = berr_rel
            if y_model is None:
                measure_detach = data_detach(self.measure)
                self.y_model = data_attach(self.predict(measure_detach['x']),
                                           measure_detach['dims'])
                self.error = mean_squared_error(self.measure, self.y_model)
                self.errorrel = 100 * (self.error / self.measure.values.mean())
            else:
                self.y_model = y_model

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
        if os.path.isfile(self.modelexecparams['parsecpydatafilepath']):
            pd = ParsecData(self.modelexecparams['parsecpydatafilepath'])
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
        phy_model = self.loadcode(self.modelcodesource, 'phymodel')
        pred = phy_model.model(self.sol, x, self.modelexecparams['overhead'])
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
        measure_detach = data_detach(self.measure)
        scores = cross_validate(ModelEstimator(self.modelexecparams,
                                verbosity=self.modelexecparams['verbosity']),
                                measure_detach['x'],
                                measure_detach['y'],
                                cv=kf, scoring=scoring['type'],
                                return_train_score=False,
                                fit_params={'dims': measure_detach['dims']},
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
        pkgname = os.path.basename(self.modelexecparams['parsecpydatafilepath'])
        pkgname = pkgname.split('_')[0]
        if not os.path.isdir(self.modelresultsfolder):
            os.mkdir(self.modelresultsfolder)
        filename = os.path.join(self.modelresultsfolder,
                                '%s_%smodel_datafile_%s.dat' % (pkgname,
                                    self.modelexecparams['algorithm'],
                                    filedate))
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
            if self.modelexecparams['algorithm'] in ['pso', 'csa']:
                datatosave['config']['modelcodesource'] = self.modelcodesource
                mep = deepcopy(self.modelexecparams)
                mep['lowervalues'] = str(mep['lowervalues'])
                mep['uppervalues'] = str(mep['uppervalues'])
                datatosave['config']['modelexecparams'] = mep
                datatosave['data']['params'] = str(list(self.sol))
                datatosave['data']['error'] = self.error
                datatosave['data']['errorrel'] = self.errorrel
                datatosave['data']['parsecdata'] = self.measure.to_dict()
                datatosave['data']['speedupmodel'] = self.y_model.to_dict()
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
            else:
                mep = deepcopy(self.modelexecparams)
                mep['c_grid'] = str(mep['c_grid'])
                mep['gamma_grid'] = str(mep['gamma_grid'])
                datatosave['config']['modelexecparams'] = mep
                datatosave['data']['error'] = self.error
                datatosave['data']['errorrel'] = self.errorrel
                datatosave['data']['parsecdata'] = self.measure.to_dict()
                datatosave['data']['speedupmodel'] = self.y_model.to_dict()
            if hasattr(self, 'measuresfraction'):
                datatosave['config']['measuresfraction'] = \
                    self.measuresfraction
                datatosave['config']['measuresfraction_points'] = \
                    self.measuresfraction_points.tolist()
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
            if configdict['modelexecparams']['algorithm'] in ['pso','csa']:
                if 'modelcodesource' in configdict.keys():
                    self.modelcodesource = configdict['modelcodesource']
                if 'modelexecparams' in configdict.keys():
                    mep = deepcopy(configdict['modelexecparams'])
                    self.modelexecparams = deepcopy(mep)
                    self.modelexecparams['lowervalues'] = json.loads(mep['lowervalues'])
                    self.modelexecparams['uppervalues'] = json.loads(mep['uppervalues'])
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
            else:
                if 'modelexecparams' in configdict.keys():
                    mep = deepcopy(configdict['modelexecparams'])
                    self.modelexecparams = deepcopy(mep)
                    self.modelexecparams['c_grid'] = json.loads(mep['c_grid'])
                    self.modelexecparams['gamma_grid'] = json.loads(mep['gamma_grid'])
            if 'measuresfraction' in configdict.keys():
                self.measuresfraction = configdict['measuresfraction']
                self.measuresfraction_points = \
                    np.array(configdict['measuresfraction_points'])
            if 'error' in datadict.keys():
                self.error = datadict['error']
            if 'errorrel' in datadict.keys():
                self.errorrel = datadict['errorrel']
            if 'parsecdata' in datadict.keys():
                self.measure = xr.DataArray.from_dict(datadict['parsecdata'])
            if 'speedupmodel' in datadict.keys():
                self.y_model = xr.DataArray.from_dict(datadict['speedupmodel'])
            if 'savedate' in configdict.keys():
                self.savedate = configdict['savedate']
        else:
            print('Error: File not found')
            return
        return

    def plot3D(self, train_points=None,
               title='Speedup Model', greycolor=False, color=False,
               showmeasures=False, linewidth=0.3, alpha=1.0, filename=''):
        """
        Plot the 3D (Speedup x cores x input size) surface.

        :param title: Plot Title.
        :param greycolor: If set color of graph to grey colormap.
        :param color: Color charactere or False
        :param linewidth: width of surface line
        :param alpha: alpha channel
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
            appname = self.measure.attrs['pkg']
            plt.title('%s\n%s' % (appname.capitalize() or None, title))
            if color:
                surf1 = ax.plot_surface(Y, X, Z, label='Model', color=color,
                                        linewidth=linewidth,
                                        edgecolor=color, linestyle='-',
                                        alpha=alpha)
            else:
                if greycolor:
                    colormap = cm.Greys
                    surf1 = ax.plot_surface(Y, X, Z, label='Model', cmap=colormap,
                                            linewidth=linewidth, edgecolor='k',
                                            linestyle='-', alpha=alpha)
                else:
                    colormap = cm.coolwarm
                    surf1 = ax.plot_surface(Y, X, Z, label='Model', cmap=colormap,
                                            linewidth=linewidth, edgecolor='r',
                                            linestyle='-', alpha=alpha)
            surf1._edgecolors2d = surf1._edgecolors3d
            surf1._facecolors2d = surf1._facecolors3d
            ax.set_xlabel(xc_label)
            if xc_label is 'Frequency':
                ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='Hz'))
            ax.set_ylabel('Number of Cores')
            ax.set_zlabel('Speedup')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            ax.set_zlim(zmin, 1.10 * zmax)
            if showmeasures:
                data_m = self.measure
                ax = fig.gca(projection='3d')
                if 'size' in data_m.dims:
                    xc = data_m.coords['size'].values
                elif 'frequency' in data_m.dims:
                    xc = 1000*data_m.coords['frequency'].values
                    if train_points is not None:
                        x_train_points = train_points.copy()
                        x_train_points[:, 0] = 1000 * x_train_points[:,0]
                yc = data_m.coords['cores'].values
                X, Y = np.meshgrid(yc, xc)
                Z = data_m.values
                surf2 = ax.plot_wireframe(Y, X, Z, linewidth=0.5,
                                          edgecolor='k')
                x = np.repeat(xc, len(yc))
                y = np.resize(yc, len(xc)*len(yc))
                z = Z.flatten()
                c = ['k' for _ in range(len(z))]
                if train_points is not None:
                    colors_index = []
                    for i in x_train_points:
                        coord_row = xc.tolist().index(i[0])
                        coord_col = yc.tolist().index(i[1])
                        colors_index.append(coord_row*len(yc)+coord_col)
                    for i in colors_index:
                        c[i] = 'lime'
                ax.scatter(x, y, z, c=c, s=3, label='Measures')
                # ax.scatter(x, y, z, c=c, s=6, label='Measures')
                # if train_points is not None:
                #     ax.scatter(xc_train_points,
                #                train_points['x'][:,1],
                #                train_points['y'],
                #                c='y', s=6)
                ax.set_zlim(min(zmin, Z.min()), 1.10 * max(zmax, Z.max()))
            ax.legend()
            ax.view_init(azim=-35, elev=28)
            if filename:
                plt.savefig(filename, dpi=1000)
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

    def __init__(self, parameters, verbosity=0):
        self.parameters = parameters
        self.model = {
            'solution': None,
            'error': None,
            'modelcodesource': None
        }
        self.verbosity = verbosity

    def fit(self, x, y, dims, **kwargs):
        """
        method to train of model with train splited data.

        :param x: measured inputs array
        :param y: measured outputs array
        :return: return ModelEstimator object.
        """

        x, y = check_X_y(x, y)
        if self.verbosity > 2:
            print('\nFit: x lenght = ', x.shape, ' y lenght = ', y.shape)
            print('x :')
            print(x)
            print('y :')
            print(y)
        kwargsmodel = {'overhead': self.parameters['overhead']}

        if self.parameters['algorithm'] == 'pso':
            optm = Swarm(self.parameters['lowervalues'],
                         self.parameters['uppervalues'],
                         parsecpydatafilepath=self.parameters['parsecpydatafilepath'],
                         modelcodefilepath=self.parameters['modelcodefilepath'],
                         size=self.parameters['size'], w=self.parameters['w'],
                         c1=self.parameters['c1'], c2=self.parameters['c2'],
                         maxiter=self.parameters['maxiter'],
                         threads=self.parameters['threads'],
                         x_meas=x, y_meas=y, verbosity=self.verbosity,
                         kwargs=kwargsmodel)
        elif self.parameters['algorithm'] == 'csa':
            initial_state = np.array([np.random.uniform(size=self.parameters['dimension'])
                                      for _ in range(self.parameters['size'])])
            optm = CoupledAnnealer(initial_state,
                                   parsecpydatafilepath=self.parameters['parsecpydatafilepath'],
                                   modelcodefilepath=self.parameters['modelcodefilepath'],
                                   size=self.parameters['size'],
                                   steps=self.parameters['steps'],
                                   update_interval=self.parameters['update_interval'],
                                   tgen_initial=self.parameters['tgen_initial'],
                                   tgen_upd_factor=self.parameters['tgen_upd_factor'],
                                   tacc_initial=self.parameters['tacc_initial'],
                                   alpha=self.parameters['alpha'],
                                   desired_variance=self.parameters['desired_variance'],
                                   lowervalues=self.parameters['lowervalues'],
                                   uppervalues=self.parameters['uppervalues'],
                                   threads=self.parameters['threads'],
                                   x_meas=x, y_meas=y, verbosity=self.verbosity,
                                   kwargs=kwargsmodel)
        else:
            print('Error: You should inform the correct algorithm to use')
            return

        self.model['error'], self.model['solution'] = optm.run()
        self.model['modelcodesource'] = optm.modelcodesource
        self.x_ = x
        self.y_ = y
        return self

    def predict(self, x):
        """
        method to test of model with test divided data.

        :param x: inputs array
        :return: return model outputs array.
        """

        if self.model['solution'] is None:
            print('Error: Model should be fitted before make predictions!')
            return
        check_is_fitted(self, ['x_', 'y_'])
        x = check_array(x)
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))

        import types
        module = types.ModuleType('physicalmodel')
        exec(self.model['modelcodesource'], module.__dict__)
        ypred = module.model(self.model['solution'], x,
                             self.parameters['overhead'])['y']
        if self.verbosity > 2:
            print('y :')
            print(ypred)
        return ypred

    @staticmethod
    def mse_score(y, y_pred, **kwargs):
        """
        method to caclculate the "Mean Squared Error" score to use
        on cross validation process.

        :param y: measured outputs array
        :param y_pred: model outputs array
        :return: return caclulated score.
        """

        mse = mean_squared_error(y, y_pred)
        return mse

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

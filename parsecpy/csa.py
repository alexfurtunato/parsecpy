# -*- coding: utf-8 -*-
"""
    Module with Classes that modeling an application using CSA( Coupled
    Simulated Annealing) based on data output generate from ParsecData Class.

    Based on PyCSA: https://github.com/structurely/csa

"""

import math
import multiprocessing as mp
import random
import time
import sys
import os
import importlib
from functools import partial
import numpy as np
import xarray as xr
import json
from copy import deepcopy
from datetime import datetime

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import ParsecData
from parsecpy import data_attach, data_detach

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    support3d = False


class CoupledAnnealer(object):
    """
    Class for performing coupled simulated annealing process.

        Attributes

      - objective_function: function
            A function which outputs a float.

      - probe_function: function
            a function which will randomly "probe"
            out from the current state, i.e. it will randomly adjust the input
            parameters for the `target_function`.

      - n_annealers: int
            The number of annealing processes to run.

      - initial_state: list
            A list of objects of length `n_probes`.
            This is used to set the initial values of the input parameters for
            `objective_function` for all `n_probes` annealing processes.

      - steps: int
            The total number of annealing steps.

      - update_interval: int
            Specifies how many steps in between updates
            to the generation and acceptance temperatures.

      - tgen_initial: float
            The initial value of the generation temperature.

      - tgen_schedule: float
            Determines the factor that tgen is multiplied by during each
            update.

      - tacc_initial: float
            The initial value of the acceptance temperature.

      - tacc_schedule: float
            Determines the factor that `tacc` is multiplied by during each
            update.

      - desired_variance: float
            The desired variance of the acceptance probabilities. If not
            specified, `desired_variance` will be set to

            :math:`0.99*(\\text{max variance}) = 0.99 * \\frac{(m - 1)}{m^2}`,

            where m is the number of annealing processes.

      - verbose: int
            Set verbose=2, 1, or 0 depending on how much output you wish to see
            (2 being the most, 0 being no output).

      - threads: int
            The number of parallel processes. Defaults to a single process.
            If `threads` <= 0, then the number of processes will be set to the
            number of available CPUs. Note that this is different from the
            `n_annealers`. If `objective_function` is costly to compute, it
            might make sense to set `n_annealers` = `processes` = max number
            of CPUs.

            On the other hand, if `objective_function` is easy to compute,
            then the CSA process will likely run a LOT faster with a single
            process due to the overhead of using multiple processes.
    """

    def __init__(self, initial_state,
                 parsecpydatapath=None,
                 modelcodepath=None,
                 modelcodesource=None,
                 n_annealers=10,
                 steps=10000,
                 update_interval=100,
                 tgen_initial=0.1,
                 tgen_upd_factor=0.99999,
                 tacc_initial=0.9,
                 alpha=0.05,
                 desired_variance=None,
                 pxmin=None,
                 pxmax=None,
                 threads=1,
                 verbosity=0,
                 args=(),
                 kwargs={}):
        self.steps = steps
        self.m = n_annealers
        self.threads = threads if threads > 0 else mp.cpu_count()
        self.update_interval = update_interval
        self.verbosity = verbosity
        self.tgen = tgen_initial
        self.tgen_upd_factor = tgen_upd_factor
        self.tacc = tacc_initial
        self.alpha = alpha
        self.pxmin = np.array(pxmin)
        self.pxmax = np.array(pxmax)
        self.args = args
        self.kwargs = kwargs
        self.tgen_initial = tgen_initial
        self.parsecpydatapath = parsecpydatapath
        self.modelcodepath = modelcodepath
        self.modelcodesource = modelcodesource
        self.best_energies = []
        self.best_states = []

        if self.modelcodepath is not None:
            pythonfile = os.path.basename(modelcodepath)
            pythonmodule = pythonfile.split('.')[0]
            if not os.path.dirname(modelcodepath):
                sys.path.append('.')
            else:
                sys.path.append(os.path.dirname(modelcodepath))
            self.modelfunc = importlib.import_module(pythonmodule)
        else:
            if modelcodesource is not None:
                import types

                self.modelfunc = types.ModuleType('csamodel')
                exec(self.modelcodesource, self.modelfunc.__dict__)

        # Set desired_variance.
        if desired_variance is None:
            self.desired_variance = 0.99 * (self.m - 1) / (self.m ** 2)
        else:
            self.desired_variance = desired_variance

        # Initialize state.
        assert len(initial_state) == self.m
        self.probe_states = initial_state.copy()

        # Shallow copy.
        self.current_states = initial_state.copy()

        # Initialize energies.
        self.probe_energies = np.zeros(self.m)
        self.current_energies = self.probe_energies.copy()

        self.probe_function = partial(self._probe_wrapper,
                                      self.modelfunc.probe_function,
                                      self.args, self.kwargs, self.tgen)
        self.objective_function = partial(self._obj_wrapper,
                                          self.modelfunc.objective_function,
                                          self.args, self.kwargs)

    def state_adjust(self, state):
        """
        Adjust limits of states from CSA (-1.0 : 1.0) to a new one.

        :param state: state to adjust.
        :return: return the adjusted state.
        """
        if len(self.pxmin)>0 and len(self.pxmax)>0:
            return self.pxmin + (self.pxmax-self.pxmin)*(state/2 + 0.5)
        return state

    @staticmethod
    def _obj_wrapper(func, args, kwargs, p):
        """
        Wrapper function that point to objective function.

        :param func: objective function.
        :param args: positional arguments to pass on to objective function
        :param kwargs: key arguments to pass on to objective function
        :param p: probe parameters used to calculate objective function
        :return: return the calculated objective function.
        """

        return func(p, *args, **kwargs)

    @staticmethod
    def _probe_wrapper(func, args, kwargs, tgen, p):
        """
        Wrapper function that point to constraint function.

        :param func: constraint function.
        :param args: positional arguments to pass on to constraint function
        :param kwargs: key arguments to pass on to constraint function
        :param p: annealer state used to calculate the probe parameters
        :return: A new probe solution based on tgen and a random function
        """

        return func(p, tgen, *args, **kwargs)

    def __update_state(self):
        """
        Update the current state across all annealers in parallel.
        """

        # Set up the mp pool.
        mpool = mp.Pool(processes=self.threads)
        # Put the workers to work.
        self.probe_states = np.array(mpool.map(self.probe_function,
                                               self.current_states.copy()))
        probe_temp = np.array([self.state_adjust(i) for i in self.probe_states])
        self.probe_energies = np.array(mpool.map(self.objective_function,
                                                 probe_temp.copy()))
        mpool.terminate()

    def __update_state_no_par(self):
        """
        Update the current state across all annealers sequentially.
        """

        for i in range(self.m):
            self.probe_states[i] = self.probe_function(
                self.current_states[i].copy())
            probe_temp = self.state_adjust(self.probe_states[i])
            self.probe_energies[i] = self.objective_function(
                probe_temp)

    def __step(self, k):
        """
        Perform one entire step of the CSA algorithm.
        """

        cool = True if k % self.update_interval == 0 else False

        max_energy = self.current_energies.max()
        exp_terms = np.exp((self.current_energies - max_energy) / self.tacc)
        prob_accept = exp_terms / exp_terms.sum()

        # Determine whether to accept or reject probe.
        for i in range(self.m):
            if self.probe_energies[i] < self.current_energies[i]:
                self.current_states[i] = self.probe_states[i].copy()
                self.current_energies[i] = self.probe_energies[i]
                if self.current_energies[i] < self.best_energies[i]:
                    self.best_states[i] = self.current_states[i].copy()
                    self.best_energies[i] = self.current_energies[i]
            elif prob_accept[i] > random.uniform(0, 1):
                self.current_states[i] = self.probe_states[i].copy()
                self.current_energies[i] = self.probe_energies[i]
            if self.verbosity > 2:
                print('Annealer %s: %s' % (i,self.current_states[i]))
                print("Best Result: State %s \nError: %s " % (self.best_states[i], self.best_energies[i]))

        # Update temperatures according to schedule.
        if cool:
            # Update generation temp.
            self.tgen = self.tgen_upd_factor*self.tgen

            # sigma2 = (sum(np.array(prob_accept)**2)*self.m - 1)/(self.m - 1)
            sigma2 = (sum(prob_accept**2)/self.m) - (1/self.m**2)
            if sigma2 < self.desired_variance:
                self.tacc *= (1 - self.alpha)
            else:
                self.tacc *= (1 + self.alpha)
            if self.verbosity > 2:
                print("Variance: ",sigma2)

    @staticmethod
    def __status_check(k, energy, temps=None, start_time=None):
        """
        Print updates to the user. Everybody is happy.
        """

        if start_time:
            elapsed = time.time() - start_time
            print("\nStep {:6d} - Error {:,.10f}, Elapsed time {:,.2f} secs"
                  .format(k, energy, elapsed))
        else:
            print("\nStep {:6d} - Error {:,.10f}".format(k, energy))
        if temps:
            print("  Updated acceptance temp {:,.10f}".format(temps[0]))
            print("  Updated generation temp {:,.10f}".format(temps[1]))
            print()

    def __get_best(self):
        """
        Return the optimal state so far.
        """

        energy = self.best_energies.min()
        index = np.where(self.best_energies == energy)[0][0]
        state = self.state_adjust(self.best_states[index])
        return energy, state

    def run(self):
        """
        Run the CSA annealing process.

        :return: return a ModelCoupledAnnealer object with best params found.
        """

        start_time = time.time()

        if self.threads > 1:
            update_func = self.__update_state
        else:
            update_func = self.__update_state_no_par

        update_func()
        self.current_energies = self.probe_energies.copy()
        self.best_energies = self.current_energies.copy()
        self.current_states = self.probe_states.copy()
        self.best_states = self.current_states.copy()

        # Run for `steps` or until user interrupts.
        for k in range(1, self.steps + 1):
            update_func()
            self.__step(k)

            if k % self.update_interval == 0 and self.verbosity > 2:
                temps = (self.tacc, self.tgen)
                self.__status_check(k, min(self.best_energies),
                                    temps=temps,
                                    start_time=start_time)
            elif self.verbosity > 1:
                self.__status_check(k, min(self.best_energies))

        best_energy, best_params = self.__get_best()
        y_measure = ParsecData(self.parsecpydatapath).speedups()
        y_measure_detach = data_detach(y_measure)
        y_pred = data_attach(self.modelfunc.model(best_params,
                                                  y_measure_detach['x'],
                                                  self.args[0]),
                             y_measure_detach['dims'])

        modelexecparams = {'m': self.m,
                           'steps': self.steps, 'dimension': len(best_params),
                           'threads': self.threads,
                           'tgen': self.tgen_initial, 'tacc': self.tacc,
                           'tgen_upd_factor': self.tgen_upd_factor,
                           'desired_variance': self.desired_variance,
                           'pxmin': list(self.pxmin),
                           'pxmax': list(self.pxmax),
                           'update_interval': self.update_interval,
                           'oh': self.args[0],
                           'modelcodepath': self.modelcodepath,
                           'parsecpydatapath': self.parsecpydatapath,
                           'alpha': self.alpha, 'verbosity': self.verbosity}
        self.modelbest = ModelCoupledAnnealer(bp=best_params,
                                              error=best_energy,
                                              ymeas=y_measure,
                                              ypred=y_pred,
                                              modelcodepath=self.modelcodepath,
                                              modelcodesource=self.modelcodesource,
                                              modelexecparams=modelexecparams)
        return self.modelbest


class ModelCoupledAnnealer:
    """
    Class that represent a speedup model of a parallel application using
    the CSA algorithm

        Atrributes
            params - position of best particle (model parameters)
            error - output of objective function for above position
            y_measure - speedups of parallel application using ParsecData
            y_model - speedups of model

        Methods
            loadata()
            savedata()
            times()
            speedups()
            plot2D()
            plot3D

    """

    def __init__(self, modeldata=None, bp=None, error=None, ymeas=None,
                 ypred=None, modelcodepath=None, modelcodesource=None,
                 modelexecparams=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param bp: best parameter of CSA
        :param ymeas: output speedup model calculated by model parameters
        :param ypred: output speedup measured by ParsecData class
        """

        if modeldata:
            self.loaddata(modeldata)
        else:
            self.y_measure = ymeas
            self.y_model = ypred
            self.modelexecparams = modelexecparams
            self.modelcodesource = None
            self.validation = None
            if modelcodepath is not None:
                f = open(modelcodepath)
                self.modelcodesource = f.read()
            if modelcodesource is not None:
                self.modelcodesource = modelcodesource
            if bp is None:
                self.params = None
                self.error = None
            else:
                self.params = bp
                self.error = error
                self.errorrel = 100*(self.error/self.y_measure.values.mean())

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

    def predict(self, args):
        """
        Predict the speedup using the input values
        and based on mathematical model.

        :param args: array with number of cores and input size
                     or array of array.
        :return: return a array with speedup values.
        """

        if self.params is None:
            print('Error: You should run or load the model data '
                  'before make predictions!')
            return None
        if not isinstance(args, np.ndarray):
            print('Error: The inputs should be a array (number of cores, '
                  'input size) or a array of array')
            return None
        if len(args.shape) == 1:
            args = args.reshape((1, args.shape[0]))
        csamodel = self.loadcode(self.modelcodesource, 'csamodel')
        y = csamodel.model(self.params, args, self.modelexecparams['oh'])
        return y

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
                    'see_error': make_scorer(CSAEstimator.see_score,
                                             parameters_number=len(
                                                 self.params))
                }
            }
        y_measure_detach = data_detach(self.y_measure)
        scores = cross_validate(CSAEstimator(self,
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
        filename = '%s_csamodel_datafile_%s.dat' % (pkgname, filedate)
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
            mep['pxmin'] = str(mep['pxmin'])
            mep['pxmax'] = str(mep['pxmax'])
            datatosave['config']['modelexecparams'] = mep
            datatosave['data']['params'] = str(list(self.params))
            datatosave['data']['error'] = self.error
            datatosave['data']['errorrel'] = self.errorrel
            datatosave['data']['parsecdata'] = self.y_measure.to_dict()
            datatosave['data']['speedupmodel'] = self.y_model.to_dict()
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
                self.modelexecparams['pxmin'] = json.loads(mep['pxmin'])
                self.modelexecparams['pxmax'] = json.loads(mep['pxmax'])
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
                self.params = json.loads(datadict['params'])
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

    def plot3D(self, title='CSA Speedup Model', greycolor=False,
               showmeasures=False, alpha=1.0, filename=''):
        """
        Plot the 3D (Speedup x cores x input size) surface.

        :param title: Plot Title.
        :param greycolor: If set color of graph to grey colormap.
        :param filename: File name to save figure (eps format).
        :return:
        """

        if not support3d:
            print('Warning: No 3D plot support. Please install matplotlib '
                  'with Axes3D toolkit')
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


class CSAEstimator(BaseEstimator, RegressorMixin):
    """
    Class of estimator to use on Cross Validation process

        Atrributes
            modeldata - SwarmModel object with modelling results
            verbosity - verbosity level of run

        Methods
            fit()
            predict()
            see_score()

    """

    def __init__(self, modeldata=None, verbosity=0):
        self.modeldata = modeldata
        self.verbosity = verbosity

    def fit(self, X, y, **kwargs):
        """
        method to train of model with train splited data.

        :param X: measured inputs array
        :param y: measured outputs array
        :return: return SwarmEstimator object.
        """

        X, y = check_X_y(X, y)
        if self.verbosity > 2:
            print('\nFit: X lenght = ', X.shape, ' y lenght = ', y.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        p = deepcopy(self.modeldata.modelexecparams)
        args = (p['oh'], {'x': X,
                          'y': y,
                          'dims': self.modeldata.y_measure.dims,
                          'input_sizes': None})
        initial_state = np.array([np.random.uniform(size=p['dimension'])
                                  for _ in range(p['m'])])
        cann = CoupledAnnealer(initial_state,
                               parsecpydatapath=p['parsecpydatapath'],
                               modelcodesource=self.modeldata.modelcodesource,
                               n_annealers=p['m'],
                               steps=p['steps'],
                               update_interval=p['update_interval'],
                               tgen_initial=p['tgen'],
                               tgen_upd_factor=p['tgen_upd_factor'],
                               tacc_initial=p['tacc'],
                               alpha=p['alpha'],
                               desired_variance=p['desired_variance'],
                               pxmin=p['pxmin'],
                               pxmax=p['pxmax'],
                               threads=p['threads'],
                               verbosity=self.verbosity,
                               args=args)
        self.modeldata = cann.run()
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        method to test of model with test splited data.

        :param X: inputs array
        :return: return model outputs array.
        """

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
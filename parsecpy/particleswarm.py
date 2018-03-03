# -*- coding: utf-8 -*-
"""
    Module with Classes that modelling an application
    based on data output generate by ParsecData Class.

"""

import sys
import os
import importlib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from functools import partial

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    support3d = False


class Particle:
    """
    Class of a particle of swarm

        Atrributes
            dim - size of swarm, is that, number of particles
            lxmin - minimum values of particle position (parameters of model)
            lxmax - maximum values of particle position (parameters of model)
            vxmin - Minimum values of particles velocity
            vxmax - Maximum values of particles velocity
            pos - actual position of particle (parameters of model)
            fpos - output of model function, is that, objective value to minimize
            vel - actual velocity of particle
            bestpos - best position found for this particle
            bestfpos - best objective value found for this particle


        Methods
            setfpos() - Set a fpos attribute and if, depend of some conditions,
                set also the bestfpos and the bestpos array os parameters
            update() - update the particle velocity and the new parameters.

    """

    def __init__(self, lxmin, lxmax, vxmin, vxmax):
        """
        Create a particle object with initial position, best position, velocity,
        objective value and best objective value.

        :param lxmin: miminum value of particle position
        :param lxmax: maximum value of particle position
        :param vxmin: miminum value of particle velocity
        :param vxmax: maximum value of particle velocity
        """

        self.dim = len(lxmin)
        self.lxmin = lxmin.copy()
        self.lxmax = lxmax.copy()
        self.vxmin = vxmin
        self.vxmax = vxmax
        self.pos = lxmin + np.random.rand(self.dim)*(lxmax - lxmin)
        self.fpos = np.inf # Infinite float
        self.vel = vxmin + np.random.rand(self.dim)*(vxmax - vxmin)
        # self.bestpos = np.zeros_like(self.pos)
        self.bestpos = self.pos.copy()
        self.bestfpos = np.inf   # Infinite float

    def __str__(self):
        """
        Default output string representation of particle

        :return: specific formated string
        """

        p = 'Pos: ' + str(self.pos) + ' - F: ' + str(self.fpos) + '\n'
        bp = 'Best Pos: ' + str(self.bestpos) + ' - F: ' + str(self.bestfpos) + '\n'
        v = 'Velocidade: ' + str(self.vel)
        return p + bp + v

    def setfpos(self,value):
        """
        Set a new fpos value. And, depend of its value, set a new bestfpos
        and new bestpos of particle.

        :param value: new fpos value to set.
        :return: return a new bestfpos.
        """

        self.fpos = value
        if self.fpos < self.bestfpos:
            self.bestfpos = self.fpos
            self.bestpos = self.pos.copy()
        return self.bestfpos

    def update(self, bestparticle, w, c1, c2):
        """
        Update a particle new velocity and new position.

        :param bestparticle: actual bestparticle of swarm.
        :param w: inertial factor used to adjust the particle velocity.
        :param c1: scaling factor for particle bestpos attribute.
        :param c2: scaling factor for bestparticle bestpos attribute.
        """

        rp = np.random.rand(self.dim)
        rg = np.random.rand(self.dim)
        #self.vel = w * self.vel + c1 * rp * (self.bestpos - self.pos) + c2 * rg * (bestp.pos - self.pos)
        phi = c1+c2
        constricao = 2*w/(abs(2-phi - np.sqrt(pow(phi,2)-4*phi)))
        self.vel = constricao * (self.vel + c1 * rp * (self.bestpos - self.pos) + c2 * rg * (bestparticle.bestpos - self.pos))
        maskvl = self.vel < self.vxmin
        maskvh = self.vel > self.vxmax
        self.vel = self.vel*(~np.logical_or(maskvl, maskvh)) + self.vxmin*maskvl + self.vxmax*maskvh
        self.pos = self.pos + self.vel
        maskl = self.pos < self.lxmin
        maskh = self.pos > self.lxmax
        self.pos = self.pos*(~np.logical_or(maskl, maskh)) + self.lxmin*maskl + self.lxmax*maskh

class Swarm:
    """
    Class of particle of swarm

        Atrributes
            maxiter - Maximum number of algorithm iterations
            threads - Number of threads to calculate the objective and
                      constratint function
            args - Positional arguments passed for objective and constraint functions
            kwargs - Key arguments passed for objective and constraint functions
            pdim - Particles dimention
            pxmin - Particle minimum position values
            pxmax - Particle maximum position values
            w - Inertial factor to calculate particle velocity
            c1 - Scaling factor for particle bestpos attribute.
            c2 - Scaling factor for best particle bestpos attribute.
            vxmin - Minimum particles velocity
            vxmax - Maximum particles velocity
            size - Size of swarm (number of particles)
            particles - List within swarm particles objects
            bestparticle - Swarm best particle object

        Methods
            _obj_wrapper()
            _constraint_wrapper()
            _swarm_med()
            run()

    """

    def __init__(self, lxmin, lxmax, modelcodepath=None, modelcodesource = None, size=100, w=0.5, c1=2, c2=2, maxiter=100,
                 threads=1, verbosity=True, parsecpydatapath=None, args=(), kwargs={}):
        """
        Initialize the particles swarm calculating the initial attribute values and
        find out the initial best particle. The objective and constraint functions are
        pointed by the swarm attributes.

        :param lxmin - Particle minimum position values
        :param lxmax - Particle maximum position values
        :param modelcodepath - path of python module with model functions
        :param modelcodesource - string with python code of model functions
        :param size - Size of swarm (number of particles)
        :param w - Inertial factor to calculate particle velocity
        :param c1 - Scaling factor for particle bestpos attribute.
        :param c2- Scaling factor for best particle bestpos attribute.
        :param maxiter - Maximum number of algorithm iterations
        :param threads - Number of threads to calculate the objective and
                      constratint function
        :param args - Positional arguments passed to objective and constraint functions
        :param kwargs - Key arguments passed to objective and constraint functions
        """

        if len(lxmin) == len(lxmax):
            lxmin = np.array(lxmin)
            lxmax = np.array(lxmax)
            if not np.all(lxmin<lxmax):
                raise AssertionError()
        else:
            raise AssertionError()

        self.maxiter = maxiter
        self.threads = threads
        self.args = args
        self.kwargs = kwargs
        self.pdim = len(lxmin)
        self.pxmin = lxmin.copy()
        self.pxmax = lxmax.copy()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vxmax = 0.1*np.abs(self.pxmax - self.pxmin)
        self.vxmin = -self.vxmax
        self.size = size
        self.particles = np.array([Particle(self.pxmin,self.pxmax,self.vxmin, self.vxmax)
                          for i in range(self.size)])
        self.modelcodepath = modelcodepath
        self.modelcodesource = modelcodesource
        self.modelbest = None
        self.parsecpydatapath = parsecpydatapath
        self.verbosity = verbosity

        if not self.modelcodepath is None:
            pythonfile = os.path.basename(modelcodepath)
            pythonmodule = pythonfile.split('.')[0]
            if not os.path.dirname(modelcodepath):
                sys.path.append('.')
            else:
                sys.path.append(os.path.dirname(modelcodepath))
            self.modelfunc = importlib.import_module(pythonmodule)
        else:
            if not modelcodesource is None:
                import types

                self.modelfunc = types.ModuleType('psomodel')
                exec(self.modelcodesource, self.modelfunc.__dict__)

        self.constr = partial(self._constraint_wrapper, self.modelfunc.constraint_function, self.args, self.kwargs)

        self.obj = partial(self._obj_wrapper, self.modelfunc.objective_function, self.args, self.kwargs)
        bestfpos = np.ones(self.size)*np.inf
        newfpos = np.zeros(self.size)
        constraint = np.zeros(self.size)
        for i,p in enumerate(self.particles):
            newfpos[i] = self.obj(p)
            constraint[i] = self.constr(p)
            if constraint[i]:
                bestfpos[i] = p.setfpos(newfpos[i])
        self.bestparticle = deepcopy(self.particles[np.argmin(bestfpos)])


    def _obj_wrapper(self, func, args, kwargs, x):
        """
        Wrapper function that point to objective function.

        :param func: objective function.
        :param args: positional arguments to pass on to objective function
        :param kwargs: key arguments to pass on to objective function
        :param x: particle used to calculate objective function
        :return: return the calculated objective function.
        """

        return func(x.pos, *args, **kwargs)

    def _constraint_wrapper(self, func, args, kwargs, x):
        """
        Wrapper function that point to constraint function.

        :param func: constraint function.
        :param args: positional arguments to pass on to constraint function
        :param kwargs: key arguments to pass on to constraint function
        :param x: particle used to calculate constraint function
        :return: If this particle is feasable or not.
        """

        return func(x.pos, *args, **kwargs)

    def _swarm_med(self):
        """
        Calculate the percentual distance between the mean of particles
        positions and the best particle position.

        :return: return the calculated parcentual distance.
        """

        med = np.array([p.fpos for p in self.particles]).mean()
        if med == np.inf or self.bestparticle.fpos == np.inf:
            return np.inf
        else:
            return abs(1 - med/self.bestparticle.fpos)

    def data_build(self, data):
        """
        Build a Dataframe data argument.

        :param data: A tuple of two lists: input values and output values
        :return: Dataframe of data.
        """

        data_dict = {}
        for x, y in zip(data[0],data[1]):
            sizename = self.args[1]['input_name']+'_'+('%02d' % x[1])
            if sizename in data_dict:
                data_dict[sizename][x[0]] = y
            else:
                data_dict[sizename] = {x[0]: y}
        df = pd.DataFrame(data_dict)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, ascending=True, inplace=True)
        return df

    def run(self):
        """
        Run the iterations of swarm algorithm.

        :return: return a ModelSwarm object with best particle found.
        """

        if self.threads > 1:
            import multiprocessing
            mpool = multiprocessing.Pool(self.threads)

        gbestfpos_ant = self.bestparticle.fpos
        gbestmax = 0
        iter = 0

        sm = self._swarm_med()
        if self.verbosity>1:
            print('\nInitial Swarm - Initial Error: ', self.bestparticle.bestfpos)

        while sm > 1e-8 and gbestmax < 10 and iter < self.maxiter:
            if self.verbosity>1:
                print('Iteration: ',iter+1,' - Error: ',self.bestparticle.bestfpos)
            for p in self.particles:
                p.update(self.bestparticle,self.w,self.c1,self.c2)
            if self.threads > 1:
                newfpos = np.array(mpool.map(self.obj, self.particles))
                constraint = np.array(mpool.map(self.constr, self.particles))
            else:
                newfpos = np.zeros(self.size)
                constraint = np.zeros(self.size)
                for i, p in enumerate(self.particles):
                    newfpos[i] = self.obj(p)
                    constraint[i] = self.constr(p)
            bestfpos = np.ones(self.size) * np.inf
            for i, p in enumerate(self.particles):
                if constraint[i]:
                    bestfpos[i] = p.setfpos(newfpos[i])
            self.bestparticle = deepcopy(self.particles[np.argmin(bestfpos)])
            if gbestfpos_ant == self.bestparticle.fpos:
                gbestmax += 1
            else:
                gbestmax = 0
                gbestfpos_ant = self.bestparticle.fpos
            iter += 1
            sm = self._swarm_med()
        if self.threads > 1:
            mpool.terminate()

        y_measure = self.data_build((self.args[1]['x'],self.args[1]['y']))
        y_pred = self.data_build(self.modelfunc.model(self.bestparticle.pos, self.args[1]['x'], self.args[0]))

        pf = self.data_build(self.modelfunc.get_parallelfraction(self.bestparticle.pos, self.args[1]['x']))
        if self.args[0]:
            oh = self.data_build(self.modelfunc.get_overhead(self.bestparticle.pos, self.args[1]['x']))
        else:
            oh = False

        modelexecparams = {'pxmin': list(self.pxmin), 'pxmax': list(self.pxmax),
                           'args': self.args, 'threads': self.threads,
                           'size': self.size, 'w': self.w, 'c1': self.c1,
                           'c2': self.c2, 'maxiter': self.maxiter,
                           'modelcodepath': self.modelcodepath,
                           'parsecpydatapath': self.parsecpydatapath,
                           'verbosity': self.verbosity}
        self.modelbest = ModelSwarm(bp=self.bestparticle,ymeas=y_measure,ypred=y_pred,pf=pf,oh=oh,modelcodepath=self.modelcodepath, modelcodesource=self.modelcodesource, modelexecparams=modelexecparams)
        return self.modelbest


class ModelSwarm:
    """
    Class of results of modelling using Swarm Optimization algorithm

        Atrributes
            params - position of best particle (model parameters)
            error - model error or output of objective function
            y_measure - measured speedups dataframe
            y_model - modeled speedups dataframe
            parallelfraction - parallel fraction calculated by model
            overhead - overhead calculated by model

        Methods
            loadata()
            savedata()
            times()
            speedups()
            plot2D()
            plot3D

    """

    def __init__(self, modeldata = None, bp=None, ymeas=None, ypred=None, pf=None, oh=False, modelcodepath=None, modelcodesource=None, modelexecparams=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param modeldata: path of the model data previously saved.
        :param bp: best particle object of swarm
        :param ymeas: output speedup model calculated by model parameters
        :param ypred: output speedup measured by ParsecData class
        :param pf: the parallel fraction calculated by parameters of model.
        :param oh: the overhead calculated by parameters of model.
        :param modelcodepath: path of the python module with model coded.
        :param modelparams: Model execution used parameters.
        """

        if modeldata:
            self.loaddata(modeldata)
        else:
            self.y_measure = ymeas
            self.y_model = ypred
            self.parallelfraction = pf
            self.overhead = oh
            self.modelexecparams = modelexecparams
            self.modelcodesource = None
            if not modelcodepath is None:
                f = open(modelcodepath)
                self.modelcodesource = f.read()
            if not modelcodesource is None:
                self.modelcodesource = modelcodesource
            if not bp:
                self.params = None
                self.error = None
            else:
                self.params = bp.pos
                self.error = bp.fpos
                self.errorrel = 100*(self.error/self.y_measure.mean().mean())

    def loadCode(self, codetext, modulename):
        """
        Load a python module stored on a string.

        :param codetext: string within model alghorithm.
        :param modulename: name of module to load with python code.
        :return: return module object with model alghorithm
        """
        import types

        module = types.ModuleType(modulename)
        exec(codetext, module.__dict__)
        return module

    def predict(self, args):
        """
        Predict the speedup using the input values and based on mathematical model.

        :param args: array with number of cores and input size or array of array.
        :return: return a array within speedup values.
        """

        if self.params is None:
            print('Error: You should run or load the model data before make predictions!')
            return None
        if not isinstance(args, np.ndarray):
            print('Error: The inputs should be a array (number of cores, input size) '
                  'or a array of array')
            return None
        if len(args.shape) == 1:
            args = args.reshape((1,args.shape[0]))
        psomodel = self.loadCode(self.modelcodesource,'psomodel')
        oh = not (type(self.overhead) is bool)
        y = psomodel.model(self.params, args, oh)
        return y

    def validate(self, kfolds=3, scoring=None):
        """
        Validate the model.

        :param args: array with number of cores and input size or array of array.
        :return: return a array within scores.
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
                    'see_error': make_scorer(SwarmEstimator.see_score,
                                             parameters_number=len(self.params))
                }
            }
        scores = cross_validate(SwarmEstimator(self,
                                verbosity=self.modelexecparams['verbosity']),
                                self.modelexecparams['args'][1]['x'],
                                self.modelexecparams['args'][1]['y'],
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
        for key,value in scores.items():
            if not key in ['fit_time','score_time']:
                if '_neg_' in key:
                    value = np.negative(value)
                self.validation['scores'][key] = {
                    'value': value,
                    'description': scoring['description'][key]
                }
        return self.validation

    def savedata(self,parsecconfig):
        """
        Write to a file the model information stored on object class

        :return: saved file name
        """

        filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        pkgname = os.path.basename(self.modelexecparams['parsecpydatapath'])
        pkgname = pkgname.split('_')[0]
        filename = '%s_psomodel_datafile_%s.dat' % (pkgname,filedate)
        with open(filename, 'w') as f:
            datatosave = {'config': {}, 'data': {}}
            if 'pkg' in parsecconfig:
                datatosave['config']['pkg'] = parsecconfig['pkg']
            if 'command' in parsecconfig:
                datatosave['config']['command'] = parsecconfig['command']
            if 'hostname' in parsecconfig:
                datatosave['config']['hostname'] = parsecconfig['hostname']
            datatosave['config']['savedate'] = datetime.now().strftime(
                "%d-%m-%Y_%H:%M:%S")
            datatosave['config']['modelcodesource'] = self.modelcodesource
            mep = deepcopy(self.modelexecparams)
            mep['pxmin'] = str(mep['pxmin'])
            mep['pxmax'] = str(mep['pxmax'])
            mep['args'][1]['xtype'] = str(mep['args'][1]['x'].dtype)
            mep['args'][1]['x'] = json.dumps(mep['args'][1]['x'].tolist())
            mep['args'][1]['ytype'] = str(mep['args'][1]['y'].dtype)
            mep['args'][1]['y'] = json.dumps(mep['args'][1]['y'].tolist())
            datatosave['config']['modelexecparams'] = mep
            datatosave['data']['params'] = pd.Series(self.params).to_json()
            datatosave['data']['error'] = self.error
            datatosave['data']['errorrel'] = self.errorrel
            datatosave['data']['parsecdata'] = self.y_measure.to_json()
            datatosave['data']['speedupmodel'] = self.y_model.to_json()
            datatosave['data']['parallelfraction'] = self.parallelfraction.to_json()
            if type(self.overhead) == bool:
                datatosave['data']['overhead'] = False
            else:
                datatosave['data']['overhead'] = self.overhead.to_json()
            if self.validation:
                val = deepcopy(self.validation)
                for key,value in val['scores'].items():
                    val['scores'][key]['type'] = str(value['value'].dtype)
                    val['scores'][key]['value'] = json.dumps(value['value'].tolist())
                for key,value in val['times'].items():
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
            if 'hostname' in configdict.keys():
                self.hostname = configdict['hostname']
            if 'modelcodesource' in configdict.keys():
                self.modelcodesource = configdict['modelcodesource']
            if 'modelexecparams' in configdict.keys():
                mep = deepcopy(configdict['modelexecparams'])
                self.modelexecparams = deepcopy(mep)
                self.modelexecparams['pxmin'] = eval(mep['pxmin'])
                self.modelexecparams['pxmax'] = eval(mep['pxmax'])
                self.modelexecparams['args'] = (mep['args'][0],{})
                self.modelexecparams['args'][1]['input_name'] = \
                    mep['args'][1]['input_name']
                self.modelexecparams['args'][1]['x'] = \
                    np.array(json.loads(mep['args'][1]['x']),
                             dtype=mep['args'][1]['xtype'])
                self.modelexecparams['args'][1]['y'] = \
                    np.array(json.loads(mep['args'][1]['y']),
                             dtype=mep['args'][1]['ytype'])
            if 'validation' in datadict.keys():
                val = deepcopy(datadict['validation'])
                for key,value in val['scores'].items():
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
                self.params = pd.Series(eval(datadict['params']))
            if 'error' in datadict.keys():
                self.error = datadict['error']
            if 'errorrel' in datadict.keys():
                self.errorrel = datadict['errorrel']
            if 'parsecdata' in datadict.keys():
                self.y_measure = pd.read_json(datadict['parsecdata'])
                self.y_measure.sort_index(inplace=True)
                self.y_measure.sort_index(axis=1, ascending=True, inplace=True)
            if 'speedupmodel' in datadict.keys():
                self.y_model = pd.read_json(datadict['speedupmodel'])
                self.y_model.sort_index(inplace=True)
                self.y_model.sort_index(axis=1, ascending=True, inplace=True)
            if 'parallelfraction' in datadict.keys():
                self.parallelfraction = pd.read_json(datadict['parallelfraction'])
                self.parallelfraction.sort_index(inplace=True)
                self.parallelfraction.sort_index(axis=1, ascending=True, inplace=True)
            if 'overhead' in datadict.keys():
                if not datadict['overhead']:
                    self.overhead = datadict['overhead']
                else:
                    self.overhead = pd.read_json(datadict['overhead'])
                    self.overhead.sort_index(inplace=True)
                    self.overhead.sort_index(axis=1, ascending=True, inplace=True)
            if 'savedate' in configdict.keys():
                self.savedate = datetime.strptime(
                    configdict['savedate'], "%d-%m-%Y_%H:%M:%S")
        else:
            print('Error: File not found')
            return
        return

    def plot3D(self, title='Model Speedup', greycolor=False, filename=''):
        """
        Plot the 3D (Speedup x cores x input size) surface.

        :param title: Plot Title.
        :param greycolor: If set color of graph to grey colormap.
        :param filename: File name to save figure (eps format).
        :return:
        """

        if not support3d:
            print('Warning: No 3D plot support. Please install matplotlib with Axes3D toolkit')
            return
        data = self.y_model
        if not data.empty:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            tests = data.columns.sort_values()
            xc = [i + 1 for (i, j) in enumerate(tests)]
            yc = data.index
            X, Y = np.meshgrid(yc, xc)
            lz = []
            for i in tests:
                lz.append(data[i])
            Z = np.array(lz)
            zmin = Z.min()
            zmax = Z.max()
            plt.title(title)
            if greycolor:
                colormap = cm.Greys
            else:
                colormap = cm.coolwarm
            surf = ax.plot_surface(Y, X, Z, cmap=colormap, linewidth=0.5,
                                   edgecolor='k', linestyle='-',
                                   vmin=(zmin - (zmax - zmin) / 10),
                                   vmax=(zmax + (zmax - zmin) / 10))
            ax.set_xlabel('Input Size')
            ax.set_xlim(0, xc[-1])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.set_ylabel('Number of Cores')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(4.0))
            ax.set_ylim(0, yc.max())
            ax.set_zlabel('Speedup')
            ax.set_zlim(zmin, 1.10 * zmax)
            ax.zaxis.set_major_locator(ticker.MultipleLocator(2.0))
            if filename:
                plt.savefig(filename, format='eps', dpi=1000)
            plt.show()


class SwarmEstimator(BaseEstimator, RegressorMixin):
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

    def __init__(self, modeldata = None, verbosity=0):
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
            print('\nFit: X lenght = ', X.shape,' y lenght = ',y.shape)
            print('X :')
            print(X)
            print('y :')
            print(y)
        p = deepcopy(self.modeldata.modelexecparams)
        args = (p['args'][0], {'x': X, 'y': y, 'input_name': p['args'][1]['input_name']})
        S = Swarm(p['pxmin'], p['pxmax'], args=args, threads=p['threads'],
                  size=p['size'], w=p['w'], c1=p['c1'], c2=p['c2'],
                  maxiter =p['maxiter'], modelcodesource=self.modeldata.modelcodesource,
                  parsecpydatapath=p['parsecpydatapath'],verbosity=self.verbosity)
        self.modeldata = S.run()
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

    def see_score(y, y_pred, **kwargs):
        """
        method to caclculate the "Standard Error of Estimator" score to use on cross validation process.

        :param y: measured outputs array
        :param y_pred: model outputs array
        :return: return caclulated score.
        """

        mse = mean_squared_error(y, y_pred)
        n = len(y)
        p = kwargs['parameters_number']
        see = np.sqrt(n * mse / (n - p))
        return see
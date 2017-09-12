# -*- coding: utf-8 -*-
"""
    Module with Classes that modeling an application
    based on data output generate from ParsecData Class.

"""

import sys
import os
import importlib
from functools import partial
import numpy as np
import copy
import json
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    support3d = False

modelfunc = None

class Particle:
    """
    Class that represent a particle of swarm

        Atrributes
            dim - size of swarm, is that, number of particles of it
            lxmin - minimum values of particle position (parameters os model)
            lxmax - maximum values of particle position (parameters os model)
            vxmin - Minimum values of velocity of particles
            vxmax - Maximum values of velocity of particles
            pos - actual position (parameters of model) of particle
            fpos - output of model function, is that, objective value to minimize
            vel - actual velocity of particle
            bestpos - best position found for this particle
            bestfpos - best objective value found for this particle


        Methods
            setfpos() - Set a fpos attribute and if, depend of some conditions,
                set also the bestfpos and the bestpos array os parameters
            update() - update the velocity of particles and the
                newpos (new parameters) of them.

    """

    def __init__(self, lxmin, lxmax, vxmin, vxmax):
        """
        Create a particle object of initial position, best position, velocity,
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
        self.bestpos = np.zeros_like(self.pos)
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

    def update(self, bestparticula, w, c1, c2):
        """
        Update a particle new velocity and new position.

        :param bestparticle: actual bestparticle of swarm.
        :param w: inertial factor used to adjust the velocity of particle.
        :param c1: scaling factor for bestpos of particle.
        :param c2: scaling factor for bestpos of bestparticle.
        :return: new calculated position of particle.
        """

        rp = np.random.rand(self.dim)
        rg = np.random.rand(self.dim)
        #self.vel = w * self.vel + c1 * rp * (self.bestpos - self.pos) + c2 * rg * (bestp.pos - self.pos)
        phi = c1+c2
        constricao = 2*w/(abs(2-phi - np.sqrt(pow(phi,2)-4*phi)))
        self.vel = constricao * (self.vel + c1 * rp * (self.bestpos - self.pos) + c2 * rg * (bestparticula.bestpos - self.pos))
        maskvl = self.vel < self.vxmin
        maskvh = self.vel > self.vxmax
        self.vel = self.vel*(~np.logical_or(maskvl, maskvh)) + self.vxmin*maskvl + self.vxmax*maskvh
        self.pos = self.pos + self.vel
        maskl = self.pos < self.lxmin
        maskh = self.pos > self.lxmax
        self.pos = self.pos*(~np.logical_or(maskl, maskh)) + self.lxmin*maskl + self.lxmax*maskh

class Swarm:
    """
    Class that represent a particle of swarm

        Atrributes
            maxiter - Maximum number of iterations of algorithm
            threads - Number of threads to use to calculate the objective and
                      constratint function
            args - Positional arguments passed for objective and constraint functions
            kwargs - Key arguments passed for objective and constraint functions
            pdim - Particles dimention
            pxmin - Particle minimum position values
            pxmax - Particle maximum position values
            w - Inertial factor to calculate velocity of particle
            c1 - Scaling factor for bestpos of particle.
            c2 - Scaling factor for bestpos of best particle.
            vxmin - Minimum velocity of particles
            vxmax - Maximum velocity of particles
            size - Size of swarm (number of particles)
            particles - A list of particles objects within swarm
            bestparticle - Best particle of swarm

        Methods
            _obj_wrapper()
            _constraint_wrapper()
            _swarm_med()
            run()

    """

    def __init__(self, lxmin, lxmax, modelpath, size=100, w=0.5, c1=2, c2=2, maxiter=100,
                 threads=1, args=(), kwargs={}):
        """
        Initialize the swarm of particle calculating the initial attribute values and
        find the initial best particle. The objective and constraint functions are
        pointed by the swarm attributes.

        :param lxmin - Particle minimum position values
        :param lxmax - Particle maximum position values
        :param modelpath - path of model file python module provided by user
        :param size - Size of swarm (number of particles)
        :param w - Inertial factor to calculate velocity of particle
        :param c1 - Scaling factor for bestpos of particle.
        :param c2- Scaling factor for bestpos of best particle.
        :param maxiter - Maximum number of iterations of algorithm
        :param threads - Number of threads to use to calculate the objective and
                      constratint function
        :param args - Positional arguments passed for objective and constraint functions
        :param kwargs - Key arguments passed for objective and constraint functions
        """

        assert len(lxmin) == len(lxmax)
        lxmin = np.array(lxmin)
        lxmax = np.array(lxmax)
        assert np.all(lxmin<lxmax)

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

        global modelfunc
        pythonfile = os.path.basename(modelpath)
        pythonmodule = pythonfile.split('.')[0]
        if not os.path.dirname(modelpath):
            sys.path.append('.')
        else:
            sys.path.append(os.path.dirname(modelpath))
        modelfunc = importlib.import_module(pythonmodule)

        self.constr = partial(self._constraint_wrapper, modelfunc.constraint_function, self.args, self.kwargs)

        self.obj = partial(self._obj_wrapper, modelfunc.objective_function, self.args, self.kwargs)
        bestfpos = np.ones(self.size)*np.inf
        newfpos = np.zeros(self.size)
        constraint = np.zeros(self.size)
        for i,p in enumerate(self.particles):
            newfpos[i] = self.obj(p)
            constraint[i] = self.constr(p)
        for i,p in enumerate(self.particles):
            if constraint[i]:
                bestfpos[i] = p.setfpos(newfpos[i])
        self.bestparticle = copy.copy(self.particles[np.argmin(bestfpos)])

    def _obj_wrapper(self, func, args, kwargs, x):
        """
        wrapper function that point to objective function provided by user
        on attibutes of object class.

        :param func: objective function .
        :param args: positional arguments to pass to objective function
        :param kwargs: key arguments to pass to objective function
        :param x: particle to calculate objective function
        :return: return the calculated objective function.
        """

        return func(x, *args, **kwargs)

    def _constraint_wrapper(self, func, args, kwargs, x):
        """
        wrapper function that point to objective function provided by user
        on attibutes of object class.

        :param func: constraint function to evaluate a particle condition.
        :param args: positional arguments to pass to objective function
        :param kwargs: key arguments to pass to objective function
        :param x: particle to calculate objective function
        :return: return if this particle is feasable or not.
        """

        return func(x, *args, **kwargs)

    def _swarm_med(self):
        """
        calculate a percentual distance of mean of particles positions by
        the best particle position.

        :return: return the calculated parcentual distance.
        """

        med = np.array([p.fpos for p in self.particles]).mean()
        if med == np.inf or self.bestparticle.fpos == np.inf:
            return np.inf
        else:
            return (1 - med/self.bestparticle.fpos)

    def run(self):
        """
        Run the iterations of swarm algorithm.

        :return: return the model object of best particle found.
        """

        if self.threads > 1:
            import multiprocessing
            mpool = multiprocessing.Pool(self.threads)

        gbestfpos_ant = self.bestparticle.fpos
        gbestmax = 0
        iter = 0

        print(self.bestparticle)

        while (self._swarm_med() > 1e-5 or gbestmax < 10) and iter < self.maxiter:
            print('Best Particle Error: ',self.bestparticle.bestfpos)
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
            self.bestparticle = copy.copy(self.particles[np.argmin(bestfpos)])
            if gbestfpos_ant == self.bestparticle.fpos:
                gbestmax += 1
            else:
                gbestmax = 0
                gbestfpos_ant = self.bestparticle.fpos
            iter += 1
        if self.threads > 1:
            mpool.terminate()

        y_measure = self.args[0]
        y_pred = modelfunc.model(self.bestparticle, self.args[1:])
        y_pred.sort_index(inplace=True)
        pf = modelfunc.get_parallelfraction(self.bestparticle.pos, self.args[1:])
        if self.args[1]:
            oh = modelfunc.get_overhead(self.bestparticle.pos, self.args[1:])
        else:
            oh = False
        modelbest = Model(self.bestparticle,y_measure,y_pred,pf,oh)
        return modelbest


class Model:
    """
    Class that represent a speedup model of a parallel application using
    the Swarm Optimization algorithm

        Atrributes
            params - position of best particle (model parameters)
            error - output of objective function for above position
            y_measure - speedups of parallel application using ParsecData
            y_model - speedups of model
            parallelfraction - parallel fraction calculated by this model
            overhead - overhead part calculated by this model

        Methods
            loadata()
            savedata()
            times()
            speedups()
            plot2D()
            plot3D

    """

    def __init__(self, bp=None, ymeas=None,ypred=None, pf=None, oh=False):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param bp: best particle object of swarm
        :param ymeas: output speedup model calculated by model parameters
        :param ypred: output speedup measured by ParsecData class
        :param pf: the parallel fraction calculated by parameters of model.
        :param oh: the overhead calculated by parameters of model.
        """

        if not bp:
            self.params = None
            self.error = None
        else:
            self.params = bp.pos
            self.error = bp.fpos
        self.y_measure = ymeas
        self.y_model = ypred
        self.parallelfraction = pf
        self.overhead = oh

    def savedata(self,parsecconfig):
        """
        Write to file the caculated model information stored on object class

        :return:
        """

        filedate = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open('swarm' + '_datafile_' + filedate \
                          + '.dat', 'w') as f:
            datatosave = {'config': {}, 'data': {}}
            if 'pkg' in parsecconfig:
                datatosave['config']['pkg'] = parsecconfig['pkg']
            if 'command' in parsecconfig:
                datatosave['config']['command'] = parsecconfig['command']
            if 'hostname' in parsecconfig:
                datatosave['config']['hostname'] = parsecconfig['hostname']
            datatosave['config']['savedate'] = datetime.now().strftime(
                "%d-%m-%Y_%H:%M:%S")
            datatosave['data']['params'] = pd.Series(self.params).to_json()
            datatosave['data']['error'] = self.error
            datatosave['data']['parsecdata'] = self.y_measure.to_json()
            datatosave['data']['speedupmodel'] = self.y_model.to_json()
            datatosave['data']['parallelfraction'] = self.parallelfraction.to_json()
            if type(self.overhead) == bool:
                datatosave['data']['overhead'] = False
            else:
                datatosave['data']['overhead'] = self.overhead.to_json()
            json.dump(datatosave, f, ensure_ascii=False)
        return

    def loaddata(self, filename):
        """
        Read a file previously saved with method savedata() and initialize
        the object class dictionaries.

        :param filename: Filename with data dictionary of execution times.
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
            if 'params' in datadict.keys():
                self.params = pd.Series(eval(datadict['params']))
            if 'error' in datadict.keys():
                self.error = datadict['error']
            if 'parsecdata' in datadict.keys():
                self.y_measure = pd.read_json(datadict['parsecdata'])
            if 'speedupmodel' in datadict.keys():
                self.y_model = pd.read_json(datadict['speedupmodel'])
            if 'parallelfraction' in datadict.keys():
                self.parallelfraction = pd.read_json(datadict['parallelfraction'])
            if 'overhead' in datadict.keys():
                if not datadict['overhead']:
                    self.overhead = datadict['overhead']
                else:
                    self.overhead = pd.read_json(datadict['overhead'])
            if 'savedate' in configdict.keys():
                self.savedate = datetime.strptime(
                    configdict['savedate'], "%d-%m-%Y_%H:%M:%S")
        else:
            print('Error: File not found')
            return
        return datadict

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

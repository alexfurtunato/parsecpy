# -*- coding: utf-8 -*-
"""
    Module with Classes that modelling an application
    based on data output generate by ParsecData Class.

"""

import sys
import os
import importlib
import numpy as np
from copy import deepcopy
from functools import partial


class Particle:
    """
    Class of a particle of swarm

        Atrributes
            dim - size of swarm, is that, number of particles
            lowervalues - minimum values of particle position (parameters of model)
            uppervalues - maximum values of particle position (parameters of model)
            vxmin - Minimum values of particles velocity
            vxmax - Maximum values of particles velocity
            pos - actual position of particle (parameters of model)
            fpos - output of model function, is that, objective
                   value to minimize
            vel - actual velocity of particle
            bestpos - best position found for this particle
            bestfpos - best objective value found for this particle


        Methods
            setfpos() - Set a fpos attribute and if, depend of some conditions,
                set also the bestfpos and the bestpos array os parameters
            update() - update the particle velocity and the new parameters.

    """

    def __init__(self, lowervalues, uppervalues, vxmin, vxmax):
        """
        Create a particle object with initial position,
        best position, velocity, objective value and best objective value.

        :param lowervalues: miminum value of particle position
        :param uppervalues: maximum value of particle position
        :param vxmin: miminum value of particle velocity
        :param vxmax: maximum value of particle velocity
        """

        self.dim = len(lowervalues)
        self.lowervalues = lowervalues.copy()
        self.uppervalues = uppervalues.copy()
        self.vxmin = vxmin
        self.vxmax = vxmax
        self.pos = lowervalues + np.random.rand(self.dim)*(uppervalues - lowervalues)
        self.fpos = np.inf  # Infinite float
        self.vel = vxmin + np.random.rand(self.dim)*(vxmax - vxmin)
        # self.bestpos = np.zeros_like(self.pos)
        self.bestpos = self.pos.copy()
        self.bestfpos = np.inf  # Infinite float

    def __str__(self):
        """
        Default output string representation of particle

        :return: specific formated string
        """

        p = 'Pos: ' + str(self.pos) + ' - F: ' + str(self.fpos) + '\n'
        bp = 'Best Pos: ' + str(self.bestpos) + ' - F: ' + \
             str(self.bestfpos) + '\n'
        v = 'Velocidade: ' + str(self.vel)
        return p + bp + v

    def setfpos(self, value):
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
        phi = c1+c2
        constricao = 2*w/(abs(2-phi - np.sqrt(pow(phi, 2)-4*phi)))
        self.vel = constricao * (self.vel + c1*rp*(self.bestpos-self.pos)
                                 + c2*rg*(bestparticle.bestpos-self.pos))
        maskvl = self.vel < self.vxmin
        maskvh = self.vel > self.vxmax
        self.vel = self.vel * (~np.logical_or(maskvl, maskvh)) \
                   + self.vxmin*maskvl + self.vxmax*maskvh
        self.pos = self.pos + self.vel
        maskl = self.pos < self.lowervalues
        maskh = self.pos > self.uppervalues
        self.pos = self.pos * (~np.logical_or(maskl, maskh)) \
                   + self.lowervalues*maskl + self.uppervalues*maskh


class Swarm:
    """
    Class of particle of swarm

        Atrributes
            maxiter - Maximum number of algorithm iterations
            threads - Number of threads to calculate the objective and
                      constratint function
            args - Positional arguments passed for objective
                   and constraint functions
            kwargs - Key arguments passed for objective
                     and constraint functions
            pdim - Particles dimention
            lowervalues - Particle minimum position values
            uppervalues - Particle maximum position values
            w - Inertial factor to calculate particle velocity
            c1 - Scaling factor for particle bestpos attribute.
            c2 - Scaling factor for best particle bestpos attribute.
            vxmin - Minimum particles velocity
            vxmax - Maximum particles velocity
            size - Size of swarm (number of particles)
            particles - List with swarm particles objects
            modelcodepath -
            modelcodesource -
            modelbest -
            parsecpydatapath -
            verbosity -
            bestparticle - Swarm best particle object

        Methods
            _obj_wrapper()
            _constraint_wrapper()
            _swarm_med()
            databuild()
            run()

    """

    # TODO: simplify the list of arguments and/or eliminate the parsecpydatpath
    def __init__(self, lowervalues, uppervalues, parsecpydatapath=None,
                 modelcodepath=None, modelcodesource=None,
                 size=100, w=0.5, c1=2, c2=2, maxiter=100,
                 threads=1, verbosity=True,
                 x_meas=None, y_meas=None,
                 kwargs={}):
        """
        Initialize the particles swarm calculating the initial attribute
        values and find out the initial best particle. The objective and
        constraint functions are pointed by the swarm attributes.

        :param lowervalues - Particle minimum position values
        :param uppervalues - Particle maximum position values
        :param modelcodepath - path of python module with model functions
        :param modelcodesource - string with python code of model functions
        :param size - Size of swarm (number of particles)
        :param w - Inertial factor to calculate particle velocity
        :param c1 - Scaling factor for particle bestpos attribute.
        :param c2- Scaling factor for best particle bestpos attribute.
        :param maxiter - Maximum number of algorithm iterations
        :param threads - Number of threads to calculate the objective and
                      constratint function
        :param verbosity - Level of verbosity of algorithm execution
        :param x_meas - Input independent variables
        :param y_meas - Input dependet variable
        :param args - Positional arguments passed to objective
                      and constraint functions
        :param kwargs - Key arguments passed to objective
                        and constraint functions
        """

        if len(lowervalues) == len(uppervalues):
            lowervalues = np.array(lowervalues)
            uppervalues = np.array(uppervalues)
            if not np.all(lowervalues < uppervalues):
                raise AssertionError()
        else:
            raise AssertionError()

        self.maxiter = maxiter
        self.threads = threads
        self.kwargs = kwargs
        self.pdim = len(lowervalues)
        self.lowervalues = lowervalues.copy()
        self.uppervalues = uppervalues.copy()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vxmax = 0.1*np.abs(self.uppervalues - self.lowervalues)
        self.vxmin = -self.vxmax
        self.size = size
        self.particles = np.array([Particle(self.lowervalues, self.uppervalues,
                                            self.vxmin, self.vxmax)
                                   for _ in range(self.size)])
        self.modelcodepath = modelcodepath
        self.modelcodesource = modelcodesource
        self.modelbest = None
        self.parsecpydatapath = parsecpydatapath
        self.verbosity = verbosity
        self.x_meas = x_meas
        self.y_meas = y_meas

        if self.modelcodepath is not None:
            pythonfile = os.path.basename(modelcodepath)
            pythonmodule = pythonfile.split('.')[0]
            if not os.path.dirname(modelcodepath):
                sys.path.append('.')
            else:
                sys.path.append(os.path.dirname(modelcodepath))
            self.modelfunc = importlib.import_module(pythonmodule)
            with open(modelcodepath) as f:
                self.modelcodesource = f.read()
        elif modelcodesource is not None:
            import types

            self.modelfunc = types.ModuleType('psomodel')
            exec(self.modelcodesource, self.modelfunc.__dict__)

        self.constr = partial(self._constraint_wrapper,
                              self.modelfunc.constraint_function,
                              self.x_meas, self.kwargs)

        self.obj = partial(self._obj_wrapper,
                           self.modelfunc.objective_function,
                           self.x_meas, self.y_meas, self.kwargs)

        bestfpos = np.ones(self.size)*np.inf
        newfpos = np.zeros(self.size)
        constraint = np.zeros(self.size)
        for i, p in enumerate(self.particles):
            newfpos[i] = self.obj(p)
            constraint[i] = self.constr(p)
            if constraint[i]:
                bestfpos[i] = p.setfpos(newfpos[i])
        self.bestparticle = deepcopy(self.particles[np.argmin(bestfpos)])

    @staticmethod
    def _obj_wrapper(func, x_meas, y_meas, kwargs, p):
        """
        Wrapper function that point to objective function.

        :param func: objective function.
        :param kwargs: key arguments to pass on to objective function
        :param p: particle used to calculate objective function
        :return: return the calculated objective function.
        """

        return func(p.pos, x_meas, y_meas, **kwargs)

    @staticmethod
    def _constraint_wrapper(func, x_meas, kwargs, p):
        """
        Wrapper function that point to constraint function.

        :param func: constraint function.
        :param kwargs: key arguments to pass on to constraint function
        :param p: particle used to calculate constraint function
        :return: If this particle is feasable or not.
        """

        return func(p.pos, x_meas, **kwargs)

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

    def run(self):
        """
        Run the iterations of swarm algorithm.

        :param x_meas - Input independent variables
        :param y_meas - Input dependet variable
        :return: return a ModelSwarm object with best particle found.
        """

        if self.threads > 1:
            import multiprocessing
            mpool = multiprocessing.Pool(self.threads)

        gbestfpos_ant = self.bestparticle.fpos
        gbestmax = 0
        iteration = 0

        sm = self._swarm_med()
        if self.verbosity > 1:
            print('\nInitial Swarm - Initial Error: ',
                  self.bestparticle.bestfpos)

        while sm > 1e-8 and gbestmax < 10 and iteration < self.maxiter:
            if self.verbosity > 1:
                print('Iteration: ', iteration+1, ' - Error: ',
                      self.bestparticle.bestfpos)
            for p in self.particles:
                p.update(self.bestparticle, self.w, self.c1, self.c2)
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
            iteration += 1
            sm = self._swarm_med()
        if self.threads > 1:
            mpool.terminate()

        return (self.bestparticle.fpos, self.bestparticle.pos)

    def get_parameters(self):
        """
        Return the Swarm Parameters used to model

        :return: Swarm parameters dictionary
        """

        modelexecparams = {'algorithm': 'pso',
                           'lowervalues': list(self.lowervalues),
                           'uppervalues': list(self.uppervalues),
                           'threads': self.threads,
                           'size': self.size, 'w': self.w, 'c1': self.c1,
                           'c2': self.c2, 'maxiter': self.maxiter,
                           'overhead': self.kwargs['overhead'],
                           'modelcodepath': self.modelcodepath,
                           'parsecpydatapath': self.parsecpydatapath,
                           'verbosity': self.verbosity}
        return modelexecparams

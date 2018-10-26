# -*- coding: utf-8 -*-
"""
    Module with Classes that modeling an application using CSA( Coupled
    Simulated Annealing) based on data output generate from ParsecData Class.

    Based on PyCSA: https://github.com/structurely/csa

"""

import multiprocessing as mp
import random
import time
import sys
import os
import importlib
from functools import partial
import numpy as np


class CoupledAnnealer(object):
    """
    Class for performing coupled simulated annealing process.

        Attributes
            size - The number of annealing processes to run.
            initial_state - A list of objects of length `n_probes`. This is
                            used to set the initial values of the input
                            parameters for `objective_function` for all
                            `n_probes` annealing processes.
            steps - The total number of annealing steps.
            update_interval - Specifies how many steps in between updates
                              to the generation and acceptance temperatures.
            tgen_initial - The initial value of the generation temperature.
            tgen_upd_factor - Determines the factor that tgen is multiplied by
                            during each  update.
            tacc_initial - The initial value of the acceptance temperature.
            desired_variance - The desired variance of the acceptance
                               probabilities. If not specified,
                               `desired_variance` will be set to
                               0.99 * \\frac{(m - 1)}{m^2}`,
                               where m is the number of annealing processes.
            verbose - Set verbose=2, 1, or 0 depending on how much output you
                      wish to see (3 being the most, 0 being no output).
            threads - The number of parallel processes. Defaults to a single
                      process. If `threads` <= 0, then the number of processes
                      will be set to the number of available CPUs. Note that
                      this is different from the `size`. If
                      `objective_function` is costly to compute, it might make
                      sense to set `size` = `processes` = max number
                      of CPUs.
                      On the other hand, if `objective_function` is easy to
                      compute, then the CSA process will likely run a LOT
                      faster with a single process due to the overhead of
                      using multiple processes.

        Methods

            objective_function - A function which outputs a float.

            probe_function - A function which will randomly "probe" out from
                             the current state, i.e. it will randomly adjust
                             the input parameters for the `target_function`.

    """

    # TODO: simplify the list of arguments and/or eliminate the parsecpydatpath
    def __init__(self, initial_state,
                 parsecpydatafilepath=None,
                 modelcodefilepath=None,
                 modelcodesource=None,
                 size=10,
                 steps=10000,
                 update_interval=100,
                 tgen_initial=0.1,
                 tgen_upd_factor=0.99999,
                 tacc_initial=0.9,
                 alpha=0.05,
                 desired_variance=None,
                 lowervalues=None,
                 uppervalues=None,
                 threads=1,
                 verbosity=0,
                 x_meas = None,
                 y_meas = None,
                 kwargs={}):
        self.steps = steps
        self.size = size
        self.threads = threads if threads > 0 else mp.cpu_count()
        self.update_interval = update_interval
        self.verbosity = verbosity
        self.tgen_initial = tgen_initial
        self.tgen = tgen_initial
        self.tgen_upd_factor = tgen_upd_factor
        self.tacc_initial = tacc_initial
        self.tacc = tacc_initial
        self.alpha = alpha
        self.lowervalues = np.array(lowervalues)
        self.uppervalues = np.array(uppervalues)
        self.kwargs = kwargs
        self.tgen_initial = tgen_initial
        self.parsecpydatafilepath = parsecpydatafilepath
        self.modelcodefilepath = modelcodefilepath
        self.modelcodesource = modelcodesource
        self.best_energies = []
        self.best_states = []
        self.x_meas = x_meas
        self.y_meas = y_meas

        if self.modelcodefilepath is not None:
            pythonfile = os.path.basename(modelcodefilepath)
            pythonmodule = pythonfile.split('.')[0]
            if not os.path.dirname(modelcodefilepath):
                sys.path.append('.')
            else:
                sys.path.append(os.path.dirname(modelcodefilepath))
            self.modelfunc = importlib.import_module(pythonmodule)
            with open(modelcodefilepath) as f:
                self.modelcodesource = f.read()
        elif modelcodesource is not None:
                import types

                self.modelfunc = types.ModuleType('csamodel')
                exec(self.modelcodesource, self.modelfunc.__dict__)

        # Set desired_variance.
        if desired_variance is None:
            self.desired_variance = 0.99 * (self.size - 1) / (self.size ** 2)
        else:
            self.desired_variance = desired_variance

        # Initialize state.
        assert len(initial_state) == self.size
        self.probe_states = initial_state.copy()

        # Shallow copy.
        self.current_states = initial_state.copy()

        # Initialize energies.
        self.probe_energies = np.zeros(self.size)
        self.current_energies = self.probe_energies.copy()

        self.probe_function = partial(self._probe_wrapper,
                                      self.modelfunc.probe_function, self.tgen)
        self.objective_function = partial(self._obj_wrapper,
                                          self.modelfunc.objective_function,
                                          self.x_meas, self.y_meas,
                                          self.kwargs)

    def state_adjust(self, state):
        """
        Adjust limits of states from CSA (-1.0 : 1.0) to a new one.

        :param state: state to adjust.
        :return: return the adjusted state.
        """
        if len(self.lowervalues)>0 and len(self.uppervalues)>0:
            return self.lowervalues + (self.uppervalues-self.lowervalues)*(state/2 + 0.5)
        return state

    @staticmethod
    def _obj_wrapper(func, x_meas, y_meas, kwargs, p):
        """
        Wrapper function that point to objective function.

        :param func: objective function.
        :param args: positional arguments to pass on to objective function
        :param kwargs: key arguments to pass on to objective function
        :param p: probe parameters used to calculate objective function
        :return: return the calculated objective function.
        """

        return func(p, x_meas, y_meas, **kwargs)

    @staticmethod
    def _probe_wrapper(func, tgen, p):
        """
        Wrapper function that point to constraint function.

        :param func: constraint function.
        :param args: positional arguments to pass on to constraint function
        :param kwargs: key arguments to pass on to constraint function
        :param p: annealer state used to calculate the probe parameters
        :return: A new probe solution based on tgen and a random function
        """

        return func(p, tgen)

    def __update_state(self):
        """
        Update the current state across all size in parallel.
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
        Update the current state across all size sequentially.
        """

        for i in range(self.size):
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
        exp_terms = np.exp((self.current_energies - max_energy)/self.tacc)
        prob_accept = exp_terms / exp_terms.sum()

        # Determine whether to accept or reject probe.
        for i in range(self.size):
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

            # sigma2 = (sum(np.array(prob_accept)**2)*self.size - 1)/(self.size - 1)
            sigma2 = (sum(prob_accept**2)/self.size) - (1/self.size**2)
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

        return self.__get_best()

    def get_parameters(self):
        """
        Return the Swarm Parameters used to model

        :return: Swarm parameters dictionary
        """

        best_energy, best_params = self.__get_best()
        modelexecparams = {'algorithm': 'csa',
                           'size': self.size,
                           'steps': self.steps, 'dimension': len(best_params),
                           'threads': self.threads,
                           'tgen_initial': self.tgen_initial,
                           'tacc_initial': self.tacc_initial,
                           'tgen_upd_factor': self.tgen_upd_factor,
                           'desired_variance': self.desired_variance,
                           'lowervalues': list(self.lowervalues),
                           'uppervalues': list(self.uppervalues),
                           'update_interval': self.update_interval,
                           'overhead': self.kwargs['overhead'],
                           'modelcodefilepath': self.modelcodefilepath,
                           'parsecpydatafilepath': self.parsecpydatafilepath,
                           'alpha': self.alpha, 'verbosity': self.verbosity}

        return modelexecparams

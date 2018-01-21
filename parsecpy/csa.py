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

try:
    xrange
except NameError:
    xrange = range

modelfunc = None

class CoupledAnnealer(object):
    """
    Interface for performing coupled simulated annealing.

    **Parameters**:

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
            Determines the factor that tgen is multiplied by during each update.

      - tacc_initial: float
            The initial value of the acceptance temperature.

      - tacc_schedule: float
            Determines the factor that `tacc` is multiplied by during each update.

      - desired_variance: float
            The desired variance of the acceptance probabilities. If not specified,
            `desired_variance` will be set to

            :math:`0.99 * (\\text{max variance}) = 0.99 * \\frac{(m - 1)}{m^2}`,

            where m is the number of annealing processes.

      - verbose: int
            Set verbose=2, 1, or 0 depending on how much output you wish to see
            (2 being the most, 0 being no output).

      - threads: int
            The number of parallel processes. Defaults to a single process.
            If `threads` <= 0, then the number of processes will be set to the
            number of available CPUs. Note that this is different from the
            `n_annealers`. If `objective_function` is costly to compute, it might
            make sense to set `n_annealers` = `processes` = max number of CPUs.

            On the other hand, if `objective_function` is easy to compute, then the
            CSA process will likely run a LOT faster with a single process due
            to the overhead of using multiple processes.
    """

    def __init__(self, modelpath,
                 n_annealers=10,
                 initial_state=[],
                 steps=10000,
                 update_interval=100,
                 tgen_initial=0.01,
                 tacc_initial=0.9,
                 alpha=0.05,
                 desired_variance=None,
                 verbose=1,
                 threads=1,
                 args=(),
                 kwargs={}):
        self.steps = steps
        self.m = n_annealers
        self.threads = threads if threads > 0 else mp.cpu_count()
        self.update_interval = update_interval
        self.verbose = verbose
        self.tgen = tgen_initial
        self.tacc = tacc_initial
        self.alpha = alpha
        self.args = args
        self.kwargs = kwargs
        self.tgen_initial = tgen_initial

        global modelfunc
        pythonfile = os.path.basename(modelpath)
        pythonmodule = pythonfile.split('.')[0]
        if not os.path.dirname(modelpath):
            sys.path.append('.')
        else:
            sys.path.append(os.path.dirname(modelpath))
        modelfunc = importlib.import_module(pythonmodule)

        self.probe_function = partial(self._probe_wrapper, modelfunc.probe_function, self.args, self.kwargs)

        self.objective_function = partial(self._obj_wrapper, modelfunc.objective_function, self.args, self.kwargs)

        # Set desired_variance.
        if desired_variance is None:
            desired_variance = 0.99
        self.desired_variance = desired_variance * (self.m - 1) / (self.m ** 2)

        # Initialize state.
        assert len(initial_state) == self.m
        self.probe_states = initial_state

        # Shallow copy.
        self.current_states = self.probe_states[:]

        # Initialize energies.
        self.probe_energies = self.current_energies = [None] * self.m

    def _obj_wrapper(self, func, args, kwargs, x):
        """
        wrapper function that point to objective function provided by user
        on attibutes of object class.

        :param func: objective function .
        :param args: positional arguments to pass to objective function
        :param kwargs: key arguments to pass to objective function
        :param x: parameter to calculate objective function
        :return: return the calculated objective function.
        """

        return func(x, *args, **kwargs)

    def _probe_wrapper(self, func, args, kwargs, x, tgen):
        """
        wrapper function that point to objective function provided by user
        on attibutes of object class.

        :param func: probe function to generate a random state.
        :param args: positional arguments to pass to objective function
        :param kwargs: key arguments to pass to objective function
        :param x: parameter to calculate objective function
        :return: return a new random state.
        """

        return func(x, tgen, *args, **kwargs)

    def __update_state(self):
        """
        Update the current state across all annealers in parallel.
        """

        # Set up the mp pool.
        pool = mp.Pool(processes=self.threads)

        # Put the workers to work.
        results = []
        for i in xrange(self.m):
            pool.apply_async(worker_probe, args=(self, i,),
                             callback=lambda x: results.append(x))

        # Gather the results from the workers.
        pool.close()
        pool.join()

        # Update the states and energies from each probe.
        for res in results:
            i, energy, probe = res
            self.probe_energies[i] = energy
            self.probe_states[i] = probe

    def __update_state_no_par(self):
        """
        Update the current state across all annealers sequentially.
        """

        for i in xrange(self.m):
            i, energy, probe = worker_probe(self, i)
            self.probe_energies[i] = energy
            self.probe_states[i] = probe

    def __step(self, k):
        """
        Perform one entire step of the CSA algorithm.
        """

        cool = True if k % self.update_interval == 0 else False

        max_energy = max(self.current_energies)
        exp_terms = []

        for i in xrange(self.m):
            E = self.current_energies[i]
            exp_terms.append(math.exp((E - max_energy) / self.tacc))

        gamma = sum(exp_terms)
        prob_accept = [x / gamma for x in exp_terms]

        # Determine whether to accept or reject probe.
        for i in xrange(self.m):
            state_energy = self.current_energies[i]
            probe_energy = self.probe_energies[i]
            probe = self.probe_states[i]
            p = prob_accept[i]
            if (probe_energy < state_energy) or (random.uniform(0, 1) < p):
                self.current_energies[i] = probe_energy
                self.current_states[i] = probe

        # Update temperatures according to schedule.
        if cool:
            # Update generation temp.
            self.tgen = self.tgen_initial/k

            sigma2 = np.var(prob_accept)
            if sigma2 < self.desired_variance:
                self.tacc *= (1 - self.alpha)
            else:
                self.tacc *= (1 + self.alpha)

    def __status_check(self, k, energy, temps=None, start_time=None):
        """
        Print updates to the user. Everybody is happy.
        """

        if start_time:
            elapsed = time.time() - start_time
            print("Step {:6d}, Energy {:,.8f}, Elapsed time {:,.2f} secs"
                  .format(k, energy, elapsed))
        else:
            print("Step {:6d}, Energy {:,.8f}".format(k, energy))
        if temps:
            print("Updated acceptance temp {:,.6f}".format(temps[0]))
            print("Updated generation temp {:,.6f}".format(temps[1]))
            print()

    def __get_best(self):
        """
        Return the optimal state so far.
        """

        energy = min(self.current_energies)
        index = self.current_energies.index(energy)
        state = self.current_states[index]
        return energy, state

    def run(self):
        """
        Run the CSA annealing process.
        """

        start_time = time.time()

        if self.threads > 1:
            update_func = self.__update_state
        else:
            update_func = self.__update_state_no_par

        update_func()
        self.current_energies = self.probe_energies[:]

        # Run for `steps` or until user interrupts.
        for k in xrange(1, self.steps + 1):
            update_func()
            self.__step(k)

            if k % self.update_interval == 0 and self.verbose >= 1:
                temps = (self.tacc, self.tgen)
                self.__status_check(k, min(self.current_energies),
                                    temps=temps,
                                    start_time=start_time)
            elif self.verbose >= 2:
                self.__status_check(k, min(self.current_energies))

        best_energy, best_params = self.__get_best()
        y_measure = self.args[0]
        y_pred = modelfunc.model(best_params, self.args[1:])
        y_pred.sort_index(inplace=True)
        pf = modelfunc.get_parallelfraction(best_params, self.args[1:])
        if self.args[1]:
            oh = modelfunc.get_overhead(best_params, self.args[1:])
        else:
            oh = False
        modelbest = ModelAnnealer(best_params, best_energy, y_measure,y_pred,pf,oh)
        return modelbest


def worker_probe(annealer, i):
    """
    This is the function that will spread across different processes in
    parallel to compute the current energy at each probe.
    """

    state = annealer.current_states[i]
    probe = annealer.probe_function(state, annealer.tgen)
    energy = annealer.objective_function(probe)
    return i, energy, probe

class ModelAnnealer:
    """
    Class that represent a speedup model of a parallel application using
    the CSA algorithm

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

    def __init__(self, bp=None, error=None, ymeas=None,ypred=None, pf=None, oh=False):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param bp: best parameter of CSA
        :param ymeas: output speedup model calculated by model parameters
        :param ypred: output speedup measured by ParsecData class
        :param pf: the parallel fraction calculated by parameters of model.
        :param oh: the overhead calculated by parameters of model.
        """

        if not bp:
            self.params = None
            self.error = None
        else:
            self.params = bp
            self.error = error
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
        filename = 'csa_datafile_%s.dat' % filedate
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
        return filename

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
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.set_ylabel('Number of Cores')
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(4.0))
            ax.set_ylim(0, yc.max())
            ax.set_zlabel('Speedup')
            ax.set_zlim(0, 1.10 * zmax)
            #ax.zaxis.set_major_locator(ticker.MultipleLocator(2.0))
            if filename:
                plt.savefig(filename, format='eps', dpi=1000)
            plt.show()

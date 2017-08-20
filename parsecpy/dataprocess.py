# -*- coding: utf-8 -*-
"""
    Class to manipulate parsec execution results.

    Use: Import the module on external scripts
"""

import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import json

class ParsecData:
    """
        Class contains two attrinutes:
            config: The metadata about execution informations
            measures: Resume dictionary with all times measures

    """
    config = {}
    measures = {}

    def __str__(self):
        if not self.config:
            return 'No data'
        pkg = 'Package: '+self.config['pkg']
        dt = 'Date: '+self.config['execdate']
        command = 'Command: '+self.config['command']
        return pkg+'\n'+dt+'\n'+command

    def __init__(self,filename=None):
        if filename:
            self.loaddata(filename)
        return

    def loaddata(self,filename):
        """
        Read a runprocess result dictionary stored on a file initialize the class dictionaries.
         - Dictionary format: index=<cores>, columns=<inputsize>, values=<time1,time2,...,timeN>.

        :param filename: Filename with data dictionary of execution times.
        """
        with open(filename) as f:
            datadict = json.load(f)
        if 'config' in datadict.keys():
            self.config = datadict['config']
        else:
            print('Warning: The config data not must read')
        if 'data' in datadict.keys():
            self.measures = datadict['data']
        else:
            print('Error: No data loaded')
        return

    def times(self):
        """
        Resume all tests, grouped by size, on DataFrame.
         - Dataframe format: index=<cores>, columns=<inputsize>, values=<median of times>.

        :return: dataframe with median of measures times.
        """

        df = DataFrame()
        data = self.measures
        inputs = list(data.keys())
        inputs.sort(reverse=True)
        for inp in inputs:
            df[inp] = Series([np.median(i) for i in data[inp].values()], index=[int(j) for j in data[inp].keys()])
        df = df.sort_index()
        return df

    def speedups(self):
        """
        Resume speedups test grouped by size on a DataFrame.
         - Dataframe format: index=<cores>, columns=<inputsize>, values=<speedups>.

        :return: dataframe with speedups.
        """

        ds = DataFrame()
        data = self.times()
        if 1 not in data.index or len(data.index) < 2:
            print("Error: Do not possible calcule the speedup without single core run")
            return
        for input in data.columns:
            idx = data.index[1:]
            darr = data.loc[1, input] / data[input][data.index != 1]
            ds[input] = Series(darr, index=idx)
        ds = ds.sort_index()
        return ds

    def plot2D(self):
        """
        Plot the 2D (Speedup x Cores) linear graph.

        """

        data = self.speedups()
        fig, ax = plt.subplots()
        for test in data.columns:
            xs = data.index
            ys = data[test]
            line, = ax.plot(xs, ys, '-', linewidth=2,label='Speedup for %s' % (test))
        ax.legend(loc='lower right')
        plt.show()

    def plot3D(self):
        """
        Plot the 3D (Speedup x cores x size) graph.

        :param data: Dataframe with speedups data.
        """

        data = self.speedups()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        tests = data.columns.sort_values()
        yc = [i+1 for (i,j) in enumerate(tests)]
        xc = data.index
        X, Y = np.meshgrid(xc, yc)
        lz = []
        for i in tests:
            lz.append(data[i])
        Z = np.array(lz)
        surf = ax.plot_surface(Y, X, Z, cmap=cm.coolwarm, linewidth = 0, antialiased = False)
        plt.show()



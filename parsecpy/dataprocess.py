# -*- coding: utf-8 -*-
"""
    Module with classes that parsec execution results.


"""

import os
from datetime import datetime
import numpy as np
from pandas import DataFrame
from pandas import Series

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

import json

class ParsecData:
    """
    Class that represent parsec run measures values

        Atrributes:
            config: The metadata about execution informations
            measures: Resume dictionary with all times measures

        Methods:
            loadata():
            times():
            speedups():
            plot2D():
            plot3D:

    """
    config = {}
    measures = {}

    def __init__(self,filename=None):
        """
        Create a object empty or with loaded data from a file

        :param filename: File name within measures for load into class
        """
        if filename:
            self.loaddata(filename)
        return

    def __str__(self):
        """
        Default string output informations of object

        :return: specific formated string
        """
        if not self.config:
            return 'No data'
        pkg = 'Package: '+self.config['pkg']
        dt = 'Date: '+self.config['execdate']
        command = 'Command: '+self.config['command']
        return pkg+'\n'+dt+'\n'+command

    def loaddata(self,filename):
        """
        Read a runprocess result dictionary stored on a file and initialize
        the class dictionaries.

        :param filename: Filename with data dictionary of execution times.
        """

        if os.path.isfile(filename):
            with open(filename) as f:
                datadict = json.load(f)
            if 'config' in datadict.keys():
                self.config['pkg'] = datadict['config']['pkg']
                self.config['execdate'] = datetime.strptime(datadict['config']['execdate'],"%Y-%m-%d_%H:%M:%S")
                self.config['command'] = datadict['config']['command']
            else:
                print('Warning: The config data not must read')
            if 'data' in datadict.keys():
                self.measures = datadict['data']
            else:
                print('Warning: No data loaded')
        else:
            print('Error: File not found')
        return

    def measurebuild(self, attrs, numberofcores=None):
        """
        Resume all tests, grouped by size, on a dictionary.
          - Dictionary format: {'testname': {'coresnumber': ['timevalue', ... ], ...}, ...}

        :param attrs: Attributes to insert into dictionary.
        :param numberofcores: Number of cores used on executed process.
        :return:
        """

        if not numberofcores:
            numberofcores = attrs['cores']

        if attrs['roitime']:
            ttime = attrs['roitime']
        else:
            ttime = attrs['realtime']

        if attrs['input'] in self.measures.keys():
            if numberofcores in self.measures[attrs['input']].keys():
                self.measures[attrs['input']][numberofcores].append(ttime)
            else:
                self.measures[attrs['input']][numberofcores] = [ttime]
        else:
            self.measures[attrs['input']] = {numberofcores: [ttime]}
        return

    def times(self):
        """
        Resume all tests, grouped by size, on DataFrame.

            Dataframe format: index=<cores>, columns=<inputsize>,
                values=<median of times>.

        :return: dataframe with median of measures times.
        """

        df = DataFrame()
        data = self.measures
        inputs = list(data.keys())
        inputs.sort(reverse=True)
        for inp in inputs:
            df[inp] = Series([np.median(i) for i in data[inp].values()],
                             index=[int(j) for j in data[inp].keys()])
        df = df.sort_index()
        return df

    def speedups(self):
        """
        Resume speedups test grouped by size on a DataFrame.

            Dataframe format: index=<cores>, columns=<inputsize>,
                values=<speedups>.

        :return: dataframe with speedups.
        """

        ds = DataFrame()
        data = self.times()
        if 1 not in data.index or len(data.index) < 2:
            print("Error: Do not possible calcule the speedup without "
                  "single core run")
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
        if data:
            fig, ax = plt.subplots()
            for test in data.columns:
                xs = data.index
                ys = data[test]
                line, = ax.plot(xs, ys, '-', linewidth=2,
                                label='Speedup for %s' % (test))
            ax.legend(loc='lower right')
            plt.show()
        else:
            print('Error: Do not possible plot data without '
                  'speedups information')

    def plot3D(self):
        """
        Plot the 3D (Speedup x cores x size) graph.

        """

        data = self.speedups()
        if data:
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
            surf = ax.plot_surface(Y, X, Z, cmap=cm.coolwarm, linewidth = 0,
                                   antialiased = False)
            plt.show()
        else:
            print('Error: Do not possible plot data without '
                  'speedups information')


class ParsecLogsData(ParsecData):
    """
    Class that represent processed parsec log files

        Attributes:

        Methods:

    """
    foldername = ''
    runfiles = []
    benchmarks = []

    def __init__(self, foldername=None):
        """
        Create a logs object empty or within processed log files data

        """

        super().__init__()
        if foldername:
            self.loaddata()
        return

    def __str__(self):
        return 'Folder: ' + self.foldername + '\nProcessed Files: \n ' \
                + '\n '.join(self.runfiles)

    def loaddata(self,foldername):
        """
        Read foldername and runfiles

        :param foldername: folder to find runfiles
        :return:
        """

        if os.path.isdir(foldername):
            self.foldername = foldername
            for root, dirs, files in os.walk(foldername):
                self.runfiles = [name for name in files if
                                 name.startswith('run_')]
            if self.runfiles:
                self.runlogfilesprocess()
                self.config['pkg'] = ', '.join(self.benchmarks)
                self.config['execdate'] = datetime.now()
        else:
            print('Error: Folder name not found.')
        return

    def contentextract(self, txt):
        """
        Extract times values from a parsec log output.

        :param txt: Content text from a parsec output run.
        :return: dict with extracted values.
        """

        for l in txt.split('\n'):
            if l.strip().startswith("[PARSEC] Benchmarks to run:"):
                benchmark = l.strip().split(':')[1]
                benchmark = benchmark.strip().split('.')[1]
            elif l.strip().startswith("[PARSEC] Unpacking benchmark input"):
                input = l.strip().split("'")[1]
            if l.strip().startswith("[HOOKS] Total time spent in ROI"):
                roitime = l.strip().split(':')[-1]
            elif l.strip().startswith("real"):
                realtime = l.strip().split('\t')[-1]
            elif l.strip().startswith("user"):
                usertime = l.strip().split('\t')[-1]
            elif l.strip().startswith("sys"):
                systime = l.strip().split('\t')[-1]
        if roitime:
            roitime = float(roitime.strip()[:-1])
        else:
            roitime = None
        if realtime:
            realtime = 60 * float(realtime.strip().split('m')[0]) \
                       + float(realtime.strip().split('m')[1][:-1])
        else:
            realtime = None
        if usertime:
            usertime = 60 * float(usertime.strip().split('m')[0]) \
                       + float(usertime.strip().split('m')[1][:-1])
        else:
            usertime = None
        if systime:
            systime = 60 * float(systime.strip().split('m')[0]) \
                      + float(systime.strip().split('m')[1][:-1])
        else:
            systime = None

        return {'benchmark': benchmark, 'input': input, 'roitime': roitime,
                'realtime': realtime, 'usertime': usertime, 'systime': systime}

    def fileprocess(self, filename):
        """
        Process a parsec log file and return a dictionary with data.

        :param filename: File name to extract the contents data.
        :return: dictionary with extracted values.
        """

        f = open(filename)
        content = f.read()
        bn = os.path.basename(filename)
        parts = bn.split("_")
        cores = int(parts[1])
        dictattrs = self.contentextract(content)
        f.close()
        dictattrs['filename'] = bn
        dictattrs['cores'] = cores
        return dictattrs

    def runlogfilesprocess(self):
        """
        Process parsec log files within a folder and load data on
        class attributes

        :return:
        """

        benchmarksset = set()
        for filename in self.runfiles:
            filepath = os.path.join(self.foldername, filename)
            fattrs = self.fileprocess(filepath)
            self.measurebuild(self.measures, fattrs)
            benchmarksset.add(fattrs['benchmark'])
        self.benchmarks = list(benchmarksset)
        return

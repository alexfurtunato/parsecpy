# -*- coding: utf-8 -*-
"""
    Module with Classes with generates Pandas Dataframes
    with prossed data from Parsec applications execution.


"""

import os
from datetime import datetime
import json
import numpy as np
from pandas import DataFrame
from pandas import Series

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    support3d = False

import json

class ParsecData:
    """
    Class that store parsec run measures values

        Atrributes
            config - The metadata about execution informations
            measures - Resume dictionary with all measures times

        Methods
            loadata()
            savedata()
            times()
            speedups()
            plot2D()
            plot3D

    """

    config = {}
    measures = {}

    def __init__(self,filename=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param filename: File name that store measures
        """

        if filename:
            self.loaddata(filename)
        return

    def __str__(self):
        """
        Default output string representation of class

        :return: specific formated string
        """

        if not self.config:
            return 'No data'
        pkg = 'Package: '+ self.config['pkg']
        dt = 'Date: ' + self.config['execdate'].strftime("%d-%m-%Y_%H:%M:%S")
        command = 'Command: '+ self.config['command']
        return pkg+'\n'+dt+'\n'+command

    def loaddata(self,filename):
        """
        Read a file previously saved with method savedata() and initialize
        the object class dictionaries.

        :param filename: Filename with data dictionary of execution times.
        """

        if os.path.isfile(filename):
            with open(filename) as f:
                datadict = json.load(f)
            if 'config' in datadict.keys():
                self.config['pkg'] = datadict['config']['pkg']
                self.config['execdate'] = datetime.strptime(
                                          datadict['config']['execdate'],
                                          "%d-%m-%Y_%H:%M:%S")
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

    def savedata(self):
        """
        Write to file the measures information stored on object class

        :return:
        """

        filedatename = self.config['execdate'].strftime("%Y-%m-%d_%H:%M:%S")
        with open(self.config['pkg'] + '_datafile_' + filedatename \
                          + '.dat', 'w') as f:
            conftxt = self.config.copy()
            conftxt['execdate'] = conftxt['execdate'].strftime("%d-%m-%Y_%H:%M:%S")
            dictsave = {'config': conftxt, 'data': self.measures}
            json.dump(dictsave, f, ensure_ascii=False)
        return

    def contentextract(self, txt):
        """
        Extract times values from a parsec log file output and return a
        dictionary of data.

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


    def measurebuild(self, attrs, numberofcores=None):
        """
        Resume all tests, grouped by input sizes and number of cores,
        on a dictionary.

        Dictionary format
            {'inputsize':{'numberofcores1':['timevalue1', ... ], ... }, ...}

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
        Return a Pandas Dataframe with resume of all tests,
        grouped by input size e number of cores.

        Dataframe format
            row indexes=<number cores>
            columns indexes=<input sizes>,
            values=<median of measures times>.

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
        Return a Pandas Dataframe with speedups,
        grouped by input size e number of cores.

        Dataframe format
            row indexes=<number cores>
            columns indexes=<input sizes>,
            values=<calculated speedups>.

        :return: dataframe with calculated speedups.
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
        Plot the 2D (Speedup x Cores) lines graph.

        """

        data = self.speedups()
        if not data.empty:
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
        Plot the 3D (Speedup x cores x input size) surface.

        """

        if not support3d:
            print('Warning: No 3D plot support. Please install matplotlib with Axes3D toolkit')
            return
        data = self.speedups()
        if not data.empty:
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
    Class that store parsec run measures values obtained from
    logs files

        Atrributes
            config: The metadata about execution informations
            measures: Resume dictionary with all measures times
            foldername: Folder where was found logs files
            runfiles: List of processed files
            benchmarks: List of benchmarks applications founder on log files

        Methods:
            loadata()
            savedata()
            fileproccess()
            runlogfilesproc()
            times()
            speedups()
            plot2D()
            plot3D

    """

    foldername = ''
    runfiles = []
    benchmarks = []

    def __init__(self, foldername=None):
        """
        Create a empty object or initialized of data from files found
        in foldername

        :param foldername: Folder name that store logs files
        """

        ParsecData.__init__(self)
        if foldername:
            self.loaddata(foldername)
        return

    def __str__(self):
        """
        Default output string representation of class

        :return: specific formated string
        """

        if not self.config:
            return 'No data'
        folder = 'Folder: ' + self.foldername
        files = 'Processed Files: \n ' \
                + '\n '.join(self.runfiles)
        pkg = 'Package: ' + self.config['pkg']
        dt = 'Date: ' + self.config['execdate'].strftime("%d-%m-%Y_%H:%M:%S")
        command = 'Command: '+ self.config['command']
        return folder + '\n' + files + '\n' + pkg+'\n' + dt + '\n' + command

    def loaddata(self,foldername):
        """
        Read all logs files that found in foldername and initialize
        the object class dictionaries.

        :param filename: Filename with data dictionary of execution times.
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
                self.config['command'] = 'logsprocess folder => ' \
                                         + self.foldername
        else:
            print('Error: Folder name not found.')
        return

    def savedata(self):
        """
        Write to file the measures information stored on object class

        :return:
        """

        filedatename = self.config['execdate'].strftime("%Y-%m-%d_%H:%M:%S")
        with open('logs_' + self.foldername + '_datafile_' + filedatename
                          + '.dat', 'w') as f:
            conftxt = self.config.copy()
            conftxt['execdate'] = conftxt['execdate'].strftime("%d-%m-%Y_%H:%M:%S")
            dictsave = {'config': conftxt, 'data': self.measures}
            json.dump(dictsave, f, ensure_ascii=False)
        return

    def fileprocess(self, filename):
        """
        Process a parsec log file and return a dictionary with processed data.

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
        object class attributes

        :return:
        """

        benchmarksset = set()
        for filename in self.runfiles:
            filepath = os.path.join(self.foldername, filename)
            fattrs = self.fileprocess(filepath)
            self.measurebuild(fattrs)
            benchmarksset.add(fattrs['benchmark'])
        self.benchmarks = list(benchmarksset)
        return

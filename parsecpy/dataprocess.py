# -*- coding: utf-8 -*-
"""
    Module with Classes that generates xArray DataArray
    with processed data from Parsec applications execution.


"""

import os
from datetime import datetime
import json
import numpy as np
import xarray as xr
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Any, List

from ._common import freq_hz

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.widgets import Slider, RadioButtons

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    support3d = False


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

    def __init__(self, filename=None):
        """
        Create a empty object or initialized of data from a file saved
        with savedata method.

        :param filename: File name that store measures
        """

        self.config = {}
        self.measures = {}
        self.power = defaultdict(list)
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
        pkg = 'Package: ' + self.config['pkg']
        dt = 'Date: ' + self.config['execdate']
        command = 'Command: ' + self.config['command']
        return pkg + '\n' + dt + '\n' + command

    def loaddata(self, filename):
        """
        Read a file previously saved with method savedata() and initialize
        the object class dictionaries.

        :param filename: Filename with data dictionary of execution times.
        """

        if os.path.isfile(filename):
            with open(filename) as f:
                datadict = json.load(f)
            if 'config' in datadict.keys():
                if 'pkg' in datadict['config']:
                    self.config['pkg'] = datadict['config']['pkg']
                if 'execdate' in datadict['config']:
                    self.config['execdate'] = datadict['config']['execdate']
                if 'command' in datadict['config']:
                    self.config['command'] = datadict['config']['command']
                if 'input_sizes' in datadict['config']:
                    self.config['input_sizes'] = datadict['config']['input_sizes']
                if 'hostname' in datadict['config']:
                    self.config['hostname'] = datadict['config']['hostname']
                if 'thread_cpu' in datadict['config']:
                    self.config['thread_cpu'] =\
                        datadict['config']['thread_cpu']
            else:
                print('Warning: The config data not must read')
            if 'data' in datadict.keys():
                self.measures = datadict['data']
            else:
                print('Warning: No data loaded')
            if 'power' in datadict.keys():
                self.power = datadict['power']
        else:
            print('Error: File not found')
        return

    def savedata(self, filename=None):
        """
        Write to file the measures information stored on object class

        :param filename: Filename to save on.
        :return:
        """

        if filename is None:
            filedatename = self.config['execdate']
            filename = self.config['pkg'] + '_datafile_' + filedatename + '.dat'
        with open(filename, 'w') as f:
            conftxt = self.config.copy()
            conftxt['execdate'] = conftxt['execdate']
            dictsave = {'config': conftxt,
                        'data': self.measures,
                        'power': self.power}
            json.dump(dictsave, f, ensure_ascii=False)
        return filename

    @staticmethod
    def contentextract(txt):
        """
        Extract times values from a parsec log file output and return a
        dictionary of data.

        :param txt: Content text from a parsec output run.
        :return: dict with extracted values.
        """

        roitime = ''
        realtime = ''
        usertime = ''
        systime = ''
        for l in txt.split('\n'):
            if l.strip().startswith("[PARSEC] Benchmarks to run:"):
                benchmark = l.strip().split(':')[1]
                if benchmark.startswith("parsec"):
                    benchmark = benchmark.strip().split('.')[1]
                else:
                    benchmark = benchmark.strip()
            elif l.strip().startswith("[PARSEC] Unpacking benchmark input"):
                inputsize = l.strip().split("'")[1]
            elif l.strip().startswith("[PARSEC] No archive for input"):
                inputsize = l.strip().split("'")[1]
            elif l.strip().startswith("[HOOKS] Total time spent in ROI"):
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

        return {'benchmark': benchmark, 'input': inputsize, 'roitime': roitime,
                'realtime': realtime, 'usertime': usertime, 'systime': systime}

    def measurebuild(self, attrs, frequency=0,
                     inputsize=None, numberofcores=None):
        """
        Resume all tests, grouped by input sizes and number of cores,
        on a dictionary.

        Dictionary format
            {'inputsize':{'numberofcores1':['timevalue1', ... ], ... }, ...}

        :param attrs: Attributes to insert into dictionary.
        :param frequency: Custom CPU frequency (Mhz) at execution moment.
        :param inputsize: Input size index used on execution.
        :param numberofcores: Number of cores used on executed process.
        :return:
        """

        if numberofcores is None:
            return None
        if inputsize is None:
            inputsize = attrs['input']
        if attrs['roitime']:
            ttime = attrs['roitime']
        else:
            ttime = attrs['realtime']

        if frequency in self.measures.keys():
            if inputsize in self.measures[frequency].keys():
                if numberofcores in self.measures[frequency][inputsize].keys():
                    self.measures[frequency][inputsize][numberofcores].\
                    append(ttime)
                else:
                    self.measures[frequency][inputsize][numberofcores] = \
                        [ttime]
            else:
                self.measures[frequency][inputsize] = {numberofcores: [ttime]}
        else:
            self.measures[frequency] = {inputsize: {numberofcores: [ttime]}}
        return

    def threadcpubuild(self, source, frequency, inputsize, numberofcores):
        """
        Resume all execution threads cpu numbers, grouped by frequencies,
        input sizes and number of cores and repetitions, on a dictionary.

        Dictionary format
            {'frequency':{'inputsize':{'numberofcores1':['cpusdict1', ... ]
            , ... }}}
        :param source: Attributes to insert into dictionary.
        :param frequency: Frequency used on execution (0 if don't fixed).
        :param inputsize: Input size used on execution.
        :param numberofcores: Number of cores used on executed process.
        :return:
        """

        threadcpu = self.config['thread_cpu']
        if frequency in threadcpu.keys():
            if inputsize in threadcpu[frequency].keys():
                if numberofcores in threadcpu[frequency][inputsize].keys():
                    inputdict = threadcpu[frequency][inputsize]
                    inputdict[numberofcores].append(source)
                else:
                    threadcpu[frequency][inputsize][numberofcores] = \
                        [source]
            else:
                threadcpu[frequency][inputsize] = \
                    {numberofcores: [source]}
        else:
            threadcpu[frequency] = \
                {inputsize: {numberofcores: [source]}}
        return

    def powerbuild(self, attrs: Dict[Any, Any], keys: List[Any]):
        """
        Resume all energy sensors measurements, grouped by frequency, input
        sizes and number of cores on a dictionary.

        Dictionary format
            {"inputsize":{"numberofcores1":["timevalue1", ... ], ... }, ...}

        :param attrs: Attributes to insert into dictionary.
        :param keys: Custom CPU frequency (Mhz) at execution moment.
        :return:
        """
        keys = filter(lambda x: x is not None, keys)
        keys = map(str, keys)
        self.power[";".join(keys)].append(attrs)

    def threads(self):
        """
        Return a xArray DataArray with resume of all threads,
        grouped by frequency, input size and number of cores.

        :return: dataframe with median of measures times.
        """

        freq = []
        cpus = []
        size = []
        cores = []
        for f in sorted(self.config['thread_cpu'].keys(), key=int):
            freq.append(int(f))
            size = []
            mf = self.config['thread_cpu'][f]
            for s in sorted(mf.keys()):
                size.append(s)
                cores = []
                mfs = mf[s]
                for c in sorted(mfs.keys(), key=int):
                    cores.append(int(c))
                    cpus.append(mfs[c])
        repetitions = range(1,len(mfs[c])+1)
        cpus = np.array(cpus)
        if len(freq) == 1:
            cpus = cpus.reshape((len(size), len(cores), len(repetitions)))
            coords = [('size', size), ('cores', cores),
                      ('repetitions', repetitions)]
        else:
            if len(size) == 1:
                cpus = cpus.reshape((len(freq), len(cores), len(repetitions)))
                coords = [('frequency', freq), ('cores', cores),
                          ('repetitions', repetitions)]
            else:
                cpus = cpus.reshape((len(freq), len(size), len(cores),
                                     len(repetitions)))
                coords = [('frequency', freq), ('size', size),
                          ('cores', cores), ('repetitions', repetitions)]
        xcpus = xr.DataArray(cpus, coords=coords)
        return xcpus

    def times(self):
        """
        Return DataArray (xarray) with resume of all tests.

        DataArray format
            dims(frequency, size, cores)
            data=numpy array with median of measures times.

        :return: DataArray with median of measures times.
        """

        freq = []
        times = []
        size = []
        cores = []
        config = deepcopy(self.config)
        config.pop('thread_cpu')
        for f in sorted(self.measures.keys()):
            freq.append(int(f))
            size = []
            mf = self.measures[f]
            for s in sorted(mf.keys(), key=int):
                size.append(int(s))
                cores = []
                mfs = mf[s]
                for c in sorted(mfs.keys(), key=int):
                    cores.append(int(c))
                    times.append(np.median(mfs[c]))
        times = np.array(times)
        if len(freq) == 1:
            times = times.reshape((len(size), len(cores)))
            coords = [('size', size), ('cores', cores)]
            if freq[0] == 0:
                config['frequency'] = 'dynamic'
            else:
                config['frequency'] = 'static: %s' % (freq[0])
        else:
            if len(size) == 1:
                times = times.reshape((len(freq), len(cores)))
                coords = [('frequency', freq), ('cores', cores)]
                config['size'] = 'static: %s' % (size[0])
            else:
                times = times.reshape((len(freq), len(size), len(cores)))
                coords = [('frequency', freq), ('size', size), ('cores', cores)]
        xtimes = xr.DataArray(times, coords=coords)
        xtimes.attrs = deepcopy(config)
        return xtimes

    def speedups(self):
        """
        Return DataArray (xarray) with resume of all speedups.

        DataArray format
            dims(frequency, size, cores)
            data=numpy array with calculated speedups.

        :return: DataArray with calculated speedups.
        """

        times = self.times()
        try:
            times.sel(cores=1)
        except:
            print('Error: Time measurement for 1 core not found')
            return None
        lcores = len(times.coords['cores'])
        ldims = []
        for c in times.dims:
            ldims.append(len(times.coords[c]))
        if len(ldims) == 2:
            timesonecore = np.repeat(times.values[:, 0], lcores).reshape(tuple(ldims))
            xspeedup = (timesonecore / times)[:,1:]
        elif len(ldims) == 3:
            timesonecore = np.repeat(times.values[:, :, 0], lcores).reshape(tuple(ldims))
            xspeedup = (timesonecore / times)[:,:,1:]
        xspeedup.attrs = times.attrs
        return xspeedup

    def efficiency(self):
        """
        Return DataArray (xarray) with resume of all efficiencies.

        DataArray format
            dims(frequency, size, cores)
            data=numpy array with calculated efficiencies.

        :return: DataArray with calculated efficiencies.
        """

        speedups = self.speedups()
        xefficency = speedups/speedups.coords['cores']
        xefficency.attrs = speedups.attrs
        return xefficency

    @staticmethod
    def plot2D(data, title='', greycolor=False, filename=''):
        """
        Plot the 2D (Speedup x Cores) lines graph.

        :param data: DataArray to plot, generate by speedups(),
                     times() or efficiency().
        :param title: Plot Title.
        :param greycolor: If set color of graph to grey colormap.
        :param filename: File name to save figure (eps format).
        :return:
        """

        if not data.size == 0:
            if len(data.dims) != 2:
                print('Error: Do not possible plot 3-dimensions data')
                return
            fig, ax = plt.subplots()
            xs = data.coords['cores'].values
            if 'size' in data.dims:
                datalines = data.coords['size'].values
                #xc_label = 'Input Size'
            elif 'frequency' in data.dims:
                datalines = data.coords['frequency'].values
                #xc_label = 'Frequency'
            if greycolor:
                colors = plt.cm.Greys(
                    np.linspace(0, 1, len(datalines) + 10))
                colors = colors[::-1]
                colors = colors[:-5]
            else:
                colors = plt.cm.jet(np.linspace(0, 1, len(datalines)))
            for i, d in enumerate(datalines):
                if 'size' in data.dims:
                    ys = data.sel(size=d)
                    legendtitle= 'Sizes'
                    legendlabel = d
                elif 'frequency' in data.dims:
                    ys = data.sel(frequency=d)
                    legendtitle= 'Frequencies'
                    legendlabel = freq_hz(d*1000)
                line, = ax.plot(xs, ys, '-', linewidth=2, color=colors[i],
                                label='Speedup for %s' % legendlabel)
            ax.legend(loc='lower right', title=legendtitle)
            ax.set_xlabel('Number of Cores')
            ax.set_xlim(0, xs.max())
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
            ax.set_ylabel('Speedup')
            ax.set_ylim(0, data.max().max()+1)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            plt.title(title)
            if filename:
                plt.savefig(filename, dpi=1000)
            plt.show()
        else:
            print('Error: Do not possible plot data without '
                  'speedups information')

    @staticmethod
    def plot3D(data, slidername=None, title='Speedup Surface', zlabel='speedup',
               greycolor=False, filename=''):
        """
        Plot the 3D (Speedup x cores x input size) surface.

        :param data: DataArray to plot, generate by speedups(),
                     times() or efficiency().
        :param slidername: name of dimension of DataArray to use on slider.
        :param title: Plot Title.
        :param zlabel: Z Axis Label.
        :param greycolor: If set color of graph to grey colormap.
        :param filename: File name to save figure (eps format).
        :return:
        """

        def update_plot3D(idx):
            ax.clear()
            if idx is None:
                dataplot = data
                if 'size' in data.dims:
                    xc = data.coords['size'].values
                    xc_label = 'Input Size'
                elif 'frequency' in data.dims:
                    xc = [i*1000 for i in data.coords['frequency'].values]
                    xc_label = 'Frequency'
            else:
                if slidername is 'size':
                    dataplot = data.sel(size=int(idx))
                    xc = [i*1000 for i in dataplot.coords['frequency'].values]
                    xc_label = 'Frequency'
                elif slidername is 'frequency':
                    dataplot = data.sel(frequency=float(idx))
                    xc = dataplot.coords['size'].values
                    xc_label = 'Input Size'
            yc = dataplot.coords['cores'].values
            X, Y = np.meshgrid(yc, xc)
            Z = dataplot.values
            zmin = data.values.min()
            zmax = data.values.max()
            surfspeedup = ax.plot_surface(Y, X, Z, cmap=colormap,
                                          linewidth=0.5, edgecolor='k',
                                          linestyle='-',
                                          vmin=(zmin - (zmax - zmin) / 10),
                                          vmax=(zmax + (zmax - zmin) / 10))
            ax.tick_params(labelsize='small')
            ax.set_xlabel(xc_label)
            if xc_label is 'Frequency':
                ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='Hz'))
            ax.set_ylabel('Number of Cores')
            ax.set_zlabel(zlabel)
            ax.set_zlim(0, 1.10 * zmax)
            fig.canvas.draw_idle()


        if not support3d:
            print('Warning: No 3D plot support. Please install matplotlib '
                  'with Axes3D toolkit')
            return

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.title(title)
        if greycolor:
            colormap = cm.Greys
        else:
            colormap = cm.coolwarm

        if not data.size == 0:
            if len(data.dims) == 2:
                idx = None
            elif len(data.dims) == 3:
                if slidername in ('size','frequency'):
                    rax = plt.axes([0.01, 0.01, 0.17,
                                    len(data.coords[slidername].values)*0.04],
                                   facecolor='lightgoldenrodyellow')
                    raxtxt = [str(i) for i in
                              data.coords[slidername].values]
                    idx = str(data.coords[slidername].values[0])
                    radio = RadioButtons(rax, tuple(raxtxt))
                    for circle in radio.circles:
                        circle.set_radius(0.03)
                    radio.on_clicked(update_plot3D)
                else:
                    print('Error: Do not possible plot data with wrong '
                          'axis names')
                    return
            else:
                print('Error: Do not possible plot data with wrong '
                      'number of axis')
                return
            update_plot3D(idx)
            if filename:
                plt.savefig(filename, dpi=1000)
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
        command = 'Command: ' + self.config['command']
        return folder + '\n' + files + '\n' + pkg+'\n' + dt + '\n' + command

    def loaddata(self, foldername):
        """
        Read all logs files that found in foldername and initialize
        the object class dictionaries.

        :param foldername: Folder name with logs files data.
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
            conftxt['execdate'] =\
                conftxt['execdate'].strftime("%d-%m-%Y_%H:%M:%S")
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
        Process parsec log files with a folder and load data on
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

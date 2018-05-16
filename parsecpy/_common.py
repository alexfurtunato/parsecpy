import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.widgets import Slider, RadioButtons

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    support3d = False


def get_python_enviroment():
    try:
        p_str = get_ipython().__class__.__name__
        if p_str == 'ZMQInteractiveShell':
            return 'jupyternotebookshell'
        elif p_str == 'TerminalInteractiveShell':
            return 'ipythonshell'
        else:
            return 'unkownshell'
    except:
        return 'pythonshell'


def data_detach(data):
    """
    Detach the independent and dependent variables from DataArray.

    :param data: A xarray DataArray with data to detach
    :return: Tuple with the variables x and y.
    """

    x = []
    y = []
    data_serie = data.to_series()
    for i in data_serie.iteritems():
        x.append(i[0])
        y.append(i[1])
    xnp = np.array(x)
    ynp = np.array(y)
    return {'x': xnp, 'y': ynp, 'dims': data.dims}


def data_attach(data, dims):
    """
    Build a xarray DataArray from tuple with independent
    and dependent variables.

    :param data: A tuple of two lists: input values and output values
    :param dims: Tuple of strings with dimensions
    :return: DataArray of data.
    """

    xnp = np.array(data[0])
    ynp = np.array(data[1])
    coords = []
    shape = []
    for i,d in enumerate(dims):
        x = sorted(np.unique(xnp[:, i]), key=int)
        coords.append((d, x))
        shape.append(len(x))
    data_da = xr.DataArray(ynp.reshape(tuple(shape)), coords=coords)
    return data_da

def freq_hz(value):
    label = float(value)
    if label >= 1e9:
        label = "%.2f GHz" % (label / 1e9)
    elif label >= 1e6:
        label = "%.2f MHz" % (label / 1e6)
    elif label >= 1e3:
        label = "%.2f KHz" % (label / 1e3)
    else:
        label = "%.2f Hz" % label
    return label


def plot2D(data, title='', greycolor=False, filename=''):
    """
    Plot the 2D (Speedupx Cores) lines graph.

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
            plt.savefig(filename, format='eps', dpi=1000)
        plt.show()
    else:
        print('Error: Do not possible plot data without '
              'speedups information')
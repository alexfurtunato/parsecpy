# -*- coding: utf-8 -*-
"""
    Module with tools functions used on classes.

"""

import os
import numpy as np
import xarray as xr
from copy import deepcopy
import psutil
import errno
import argparse
import ghalton as gh

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import ticker
# from matplotlib.widgets import Slider, RadioButtons

support3d = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    support3d = False


#
# Functions to Arguments validations
#

def argsparsefraction(txt):
    """
    Validate the txt argument as value between 0.0 and 1.0.

    :param txt: argument is a float string between 0.0 and 1.0.
    :return: float
    """

    msg = "Value shoud be a float between 0.0 and 1.0"
    try:
        value = float(txt)
        if value < 0 or value > 1.0:
            raise argparse.ArgumentTypeError(msg)
        return value
    except ValueError:
        raise argparse.ArgumentTypeError(msg)


def argsparselist(txt):
    """
    Validate the list of txt argument.

    :param txt: argument with comma separated int strings.
    :return: list of strings.
    """

    txt = txt.split(',')
    listarg = [i.strip() for i in txt]
    return listarg


def argsparseintlist(txt):
    """
    Validate the list of int arguments.

    :param txt: argument with comma separated numbers.
    :return: list of integer converted numbers.
    """

    txt = txt.split(',')
    listarg = [int(i) for i in txt]
    return listarg


def argsparseinputlist(txt):
    """
    Validate the single or multiple input names argument.
     - Formats:
       - Single: one input name string. Ex: native.
       - Multiple: input names with sequential range numbers. Ex: native02:05

    :param txt: argument of input name.
    :return: list with a single input name or multiples separated input names.
    """

    inputsets = []
    if txt.count(':') == 0:
        inputsets.append(txt)
    elif txt.count(':') == 1:
        ifinal = txt.split(':')[1]
        if ifinal.isdecimal():
            ifinal = int(ifinal)
            iname = list(txt.split(':')[0])
            iinit = ''
            for i in iname[::-1]:
                if not i.isdecimal():
                    break
                iinit += iname.pop()
            if len(iinit):
                iname = ''.join(iname)
                iinit = int(iinit[::-1])
                inputsets = [iname + ('%02d' % i) for i in range(iinit,
                                                                 ifinal + 1)]
            else:
                msg = "Wrong compost inputset name syntax: \nParameter " \
                      "<initialnumber> parameter snot found. <inputsetname>_" \
                      "[<initialnumber>:<finalnumber>]. Ex: native_01:10"
                raise argparse.ArgumentTypeError(msg)
        else:
            msg = "\nWrong compost inputset name syntax: \nParameter " \
                  "<finalnumber> not found. <inputsetname>_" \
                  "[<initialnumber>:<finalnumber>]. Ex: native_01:10"
            raise argparse.ArgumentTypeError(msg)
    else:
        msg = "\nWrong compost inputset name syntax: \nYou should specify " \
              "only two input sizes. <inputsetname>_" \
              "[<initialnumber>:<finalnumber>]. \nEx: native_01:10"
        raise argparse.ArgumentTypeError(msg)
    return inputsets


def argsparsefloatlist(txt):
    """
    Validate the list int argument.

    :param txt: argument of comma separated int strings.
    :return: list of integer converted ints.
    """

    txt = txt.split(',')
    listarg = [float(i.strip()) for i in txt]
    return listarg


#
# Functions CPU processes and Threads monitoring
#


def thread_cpu_num(proc_id, thread_id):
    fname = "/proc/%s/task/%s/stat" % (proc_id, thread_id)
    try:
        with open(fname, 'rb') as f:
            st = f.read().strip()
    except IOError as err:
        if err.errno == errno.ENOENT:
            # no such file or directory; it means thread
            # disappeared on us
            pass
        raise
    st = st[st.find(b')') + 2:]
    values = st.split(b' ')
    cpu_num = int(values[36])
    return cpu_num


def find_procs_by_name(name):
    """
    Return a list of processes ids with 'name' on command line.

    :param name: Name to search on running process.
    :return: list of processes ids
    """

    ls = []
    for p in psutil.process_iter(attrs=["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)
    return ls


def procs_list(name, prs=None):
    """
    Buil a dictionary with running threads of a specific process.

    :param name: Name to search on running process.
    :param prs: threads processed before
    :return: dictionary of processed threads.
    """

    procs = find_procs_by_name(name)

    if prs is None:
        pts = {}
    else:
        pts = prs
    for p in procs:
        if p.pid in pts.keys():
            thr = deepcopy(pts[p.pid])
        else:
            thr = {}
        cpuchanged = False
        for t in p.threads():
            cpu_num = thread_cpu_num(p.pid, t.id)
            if t.id in thr.keys():
                if thr[t.id][-1] != cpu_num:
                    cpuchanged = True
                thr[t.id].append(cpu_num)
            else:
                thr[t.id] = [cpu_num]
                cpuchanged = True
        if cpuchanged:
            pts[p.pid] = deepcopy(thr)
    return pts


#
# Functions to detect python enviroment
#


def get_python_enviroment():
    """
    Detect the Python enviroment where the scripts running on.

    :return: String with name of enviroment: pythonshell, ipythonshell,
             jupyternotebookshell or unknownshell.
    """

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


#
# Functions to conversion xarray and lists
#


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

# TODO: Refactoring the attach metrhod to input unsorted data
def data_attach(data, dims):
    """
    Build a xarray DataArray from tuple with independent
    and dependent variables.

    :param data: A tuple of two lists: input values and output values
    :param dims: Tuple of strings with dimensions
    :return: DataArray of data.
    """

    xnp = np.array(data['x'])
    ynp = np.array(data['y'])
    coords = []
    shape = []
    for i, d in enumerate(dims):
        x = sorted(np.unique(xnp[:, i]), key=int)
        coords.append((d, x))
        shape.append(len(x))

    sorted_base = []
    for i in range(len(coords) - 1):
        for j in coords[i][1]:
            for w in coords[i + 1][1]:
                sorted_base.append([j, w])
    idx_base = [np.where((xnp == (f, c)).all(axis=1))[0][0] for f, c in
                sorted_base]

    data_da = xr.DataArray(ynp[idx_base].reshape(tuple(shape)), coords=coords)
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

def maptosequence(fseq,iseq):
    """
    Map a sequence of floats, each element in range 0.0-1.0,
    to an another sequence of values, find the elements whose indexes are
    equivalent to a relative position in a range 0-1.0.

    :param fseq: A list of float values
    :param iseq: A list of target values
    :return: A list of integer values equivalent to range of floats.
    """
    equiv_seq = []
    folds = len(iseq)-1
    for i in fseq:
        if i<0 or i>1.0:
            print("Error: Sequence of floats should be only values "
                  "between 0.0 and 1.0")
            return None
        equiv_seq.append(iseq[round(float(i)*folds)])
    return(equiv_seq)

def measures_split_train_test(measure, train_size):
    """
    Split the train and test arrays from a xarray of measures using the
    Halton sequence to make discrepancy less. The return object is a
    list of arrays: [train_x, teste_x, train_y, test_y]

    :param measure: A xarray os measures values
    :param train_size: A integer with sie of elements splited  to train
    :return: A list of arrays.
    """

    m_detach = data_detach(measure)
    if len(m_detach['x'])<train_size:
        print("Error: the train size shoud be lower than the size of arrays")
        return None
    dim = len(measure.dims)
    sequencer = gh.Halton(dim)
    points = np.array(sequencer.get(train_size))
    x_rand = []
    for i,v in enumerate(measure.dims):
        x = measure.coords[v].values
        x_rand.append(maptosequence(points[:,i],x))
    x_rand = np.column_stack([i.reshape(len(i), 1) for i in np.array(x_rand)])
    bool_idx = None
    for i in x_rand:
        if bool_idx is None:
            bool_idx = (m_detach['x'] == i).all(axis=1)
        else:
            bool_idx = bool_idx | (m_detach['x'] == i).all(axis=1)
    x_train = m_detach['x'][bool_idx]
    x_test = m_detach['x'][~bool_idx]
    y_train = m_detach['y'][bool_idx]
    y_test = m_detach['y'][~bool_idx]
    return [x_train,x_test,y_train,y_test]

# def plot2D(data, title='', greycolor=False, filename=''):
#     """
#     Plot the 2D (Speedupx Cores) lines graph.
#
#     :param data: DataArray to plot, generate by speedups(),
#                  times() or efficiency().
#     :param title: Plot Title.
#     :param greycolor: If set color of graph to grey colormap.
#     :param filename: File name to save figure (eps format).
#     :return:
#     """
#
#     if not data.size == 0:
#         if len(data.dims) != 2:
#             print('Error: Do not possible plot 3-dimensions data')
#             return
#         fig, ax = plt.subplots()
#         xs = data.coords['cores'].values
#         if 'size' in data.dims:
#             datalines = data.coords['size'].values
#             #xc_label = 'Input Size'
#         elif 'frequency' in data.dims:
#             datalines = data.coords['frequency'].values
#             #xc_label = 'Frequency'
#         if greycolor:
#             colors = plt.cm.Greys(
#                 np.linspace(0, 1, len(datalines) + 10))
#             colors = colors[::-1]
#             colors = colors[:-5]
#         else:
#             colors = plt.cm.jet(np.linspace(0, 1, len(datalines)))
#         for i, d in enumerate(datalines):
#             if 'size' in data.dims:
#                 ys = data.sel(size=d)
#                 legendtitle= 'Sizes'
#                 legendlabel = d
#             elif 'frequency' in data.dims:
#                 ys = data.sel(frequency=d)
#                 legendtitle= 'Frequencies'
#                 legendlabel = freq_hz(d*1000)
#             line, = ax.plot(xs, ys, '-', linewidth=2, color=colors[i],
#                             label='Speedup for %s' % legendlabel)
#         ax.legend(loc='lower right', title=legendtitle)
#         ax.set_xlabel('Number of Cores')
#         ax.set_xlim(0, xs.max())
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
#         ax.set_ylabel('Speedup')
#         ax.set_ylim(0, data.max().max()+1)
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
#         plt.title(title)
#         if filename:
#             plt.savefig(filename, dpi=1000)
#         plt.show()
#     else:
#         print('Error: Do not possible plot data without '
#               'speedups information')

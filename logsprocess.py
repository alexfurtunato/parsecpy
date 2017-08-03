#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
  Module to process parsec output informations.
  The module can be used on others modules and as a executabled script.
  As executabled script, its process a folder with log run files.
"""

import sys
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import tarfile


# Template with 2 arguments: filename.runconf, input name
template_parsecconf = '''#!/bin/bash
# %s - file containing information necessary to run a specific
#                     program of the PARSEC benchmark suite

# Description of the input set
run_desc="%s input for performance analysis with simulators"

# Citations for this input set (requires matching citation file)
pkg_cite="bienia11parsec"
'''

# Template with 3 arguments: filename.runconf, filename.dat, minimum support
template_freqmine_inputconf =  '''#!/bin/bash
# %s - file containing information necessary to run a specific
#                    program of the PARSEC benchmark suite

# Binary file to execute, relative to installation root
run_exec="bin/freqmine"

# Set number of OpenMP threads
export OMP_NUM_THREADS=${NTHREADS}

# Arguments to use
run_args="%s %s"
'''

template_fluidanimate_inputconf =  '''#!/bin/bash
#
# %s - file containing information necessary to run a specific
#                  program of the PARSEC benchmark suite
#

# Binary file to execute, relative to installation root
run_exec="bin/fluidanimate"

# Arguments to use
run_args="${NTHREADS} %s in_500K.fluid out.fluid"
'''

def contentextract(txt):
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
        realtime = 60*float(realtime.strip().split('m')[0]) + float(realtime.strip().split('m')[1][:-1])
    else:
        realtime = None
    if usertime:
        usertime = 60 * float(usertime.strip().split('m')[0]) + float(usertime.strip().split('m')[1][:-1])
    else:
        usertime = None
    if systime:
        systime = 60 * float(systime.strip().split('m')[0]) + float(systime.strip().split('m')[1][:-1])
    else:
        systime = None

    return {'benchmark': benchmark, 'input': input, 'roitime':roitime, 'realtime': realtime, 'usertime': usertime, 'systime': systime}

def fileprocess(fn):
    """
    Process a parsec log file and return a dictionary with data.

    :param fn: File name to extract the contents data.
    :return: dictionary with extracted values.
    """

    f = open(fn)
    content = f.read()
    bn = os.path.basename(fn)
    parts = bn.split("_")
    cores = int(parts[1])
    dictattrs = contentextract(content)
    f.close()
    dictattrs['filename'] = bn
    dictattrs['cores'] = cores
    return dictattrs

def datadictbuild(rdict, attrs, ncores=None):
    """
    Resume all tests, grouped by size, on a dictionary.
      - Dictionary format: {'testname': {'coresnumber': ['timevalue', ... ], ...}, ...}

    :param rdict: Dictionary to store the data attributes.
    :param attrs: Attributes to insert into dictionary.
    :param ncores: Number of cores used on executed process.
    :return: dictionary with inserted values.
    """

    if not ncores:
        ncores = attrs['cores']

    if attrs['roitime']:
        ttime = attrs['roitime']
    else:
        ttime = attrs['realtime']

    if attrs['input'] in rdict.keys():
        if ncores in rdict[attrs['input']].keys():
            rdict[attrs['input']][ncores].append(ttime)
        else:
            rdict[attrs['input']][ncores] = [ttime]
    else:
        rdict[attrs['input']] = {ncores: [ttime]}
    return rdict

def dataframebuild(data):
    """
    Resume all tests, grouped by size, on DataFrame.
     - Dataframe format: index=<cores>, columns=<inputsize>, values=<median of times>.

    :param data: Dictionary with values of calculated median of times.
    :return: dataframe with data.
    """

    df = DataFrame()
    inputs = list(data.keys())
    inputs.sort(reverse=True)
    for i in inputs:
        df[i] = Series([np.median(i) for i in data[i].values()], index=data[i].keys())
    df = df.sort_index()
    return df

def speedupframebuild(data):
    """
    Resume speedups test grouped by size on a DataFrame.
     - Dataframe format: index=<cores>, columns=<inputsize>, values=<speedups>.

    :param data: Dataframe with values of calculated speedups.
    :return: dataframe with data.
    """

    ds = DataFrame()
    if 1 not in data.index or len(data.index) < 2:
        print("Error: Do not possible calcule the speedup without single core run")
        return
    for input in data.columns:
        idx = data.index[1:]
        darr = data.loc[1, input] / data[input][data.index != 1]
        ds[input] = Series(darr, index=idx)
    ds = ds.sort_index()
    return ds

def runlogfilesprocess(fn, fl):
    """
    Process parsec log files within a folder.

    :param fn: Folder name.
    :param fl: Filename list of parsec log files.
    :return: dataframe with extracted data.
    """

    datadict = {}
    for filename in fl:
        fpath = os.path.join(fn, filename)
        fattrs = fileprocess(fpath)
        datadict = datadictbuild(datadict, fattrs)
    d = dataframebuild(datadict)
    return d

def runlogfilesfilter(fn):
    """
    Return a list of files into a folder. The filenames pattern search start with 'run_'.

    :param fn: Folder name.
    :return: list of founded files.
    """

    for root, dirs, files in os.walk(fn):
        rfiles = [name for name in files if name.startswith('run_')]
    if len(rfiles) == 0:
        return []
    return rfiles

def plot2Dspeedup(data):
    """
    Plot the 2D (Speedup x Cores) linear graph.

    :param data: Dataframe with speedups data.
    """

    print("Plotting 2D Speedup Graph... ")
    data.plot()
'''    fig, ax = plt.subplots()
    for test in data.columns:
        xs = data.index
        ys = data[test]
        line, = ax.plot(xs, ys, '-', linewidth=2,label='Speedup for %s' % (test))
    ax.legend(loc='lower right')
    plt.show()
'''

def plot3Dspeedup(data):
    """
    Plot the 3D (Speedup x cores x size) graph.

    :param data: Dataframe with speedups data.
    """

    print("Plotting 3D Speedup Graph... ")
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

def fluidanimate_inputfilesplitequal(tfile, n, frmax):
    """
    Generate Fluidanimate Benchmark inputs tar file with sames equal sizes,
    but, with different number of frames: 50, 100, 150, ... 500.

    :param tfile: Fluidanimate input tar file name to replicate.
    :param n: Count of equal replicate parts.
    :param frmax: Maximum number of frames.
    """

    prefixfolder = 'fluidanimate_inputfiles'
    parsecconffolder = os.path.join(prefixfolder,'parsecconf')
    pkgconffolder = os.path.join(prefixfolder,'pkgconf')
    os.mkdir(prefixfolder)
    os.mkdir(parsecconffolder)
    os.mkdir(pkgconffolder)
    tarfilename = os.path.basename(tfile).split('.')[0]
    print('Replicating input file %s on %s files' % (tfile,n))
    lfrm = [int(frmax - (n-1-i)*frmax/n) for i in range(n)]
    for i,frm in enumerate(lfrm):
        newtarfilename = os.path.join(prefixfolder, tarfilename + '_' + '%02d' % (i+1) + '.tar')
        print(i+1,newtarfilename)
        try:
            shutil.copy(tfile,newtarfilename)
        except OSError as why:
            print(why)
        fnbase = os.path.basename(newtarfilename).split('.')[0]
        fnbase = '_'.join(fnbase.split('_')[1:])
        fpath1 = os.path.join(parsecconffolder,fnbase+'.runconf')
        fconf1 = open(fpath1,'w')
        fconf1.write(template_parsecconf % (fnbase+'.runconf',fnbase))
        fconf1.close()
        fpath2 = os.path.join(pkgconffolder,fnbase+'.runconf')
        fconf2 = open(fpath2,'w')
        fconf2.write(template_fluidanimate_inputconf % (fnbase+'.runconf',frm))
        fconf2.close()
    print('Sucessful finished split operation. Total parts = ',i+1)


def freqmine_inputfilesplitequal(tfile, n, ms):
    """
    Split Freqmine Benchmark input tar file within 'n' equal size
    parts of new tar files with names like originalname_1 ... originalname_n.

    :param tfile: Freqmine input tar file name to split.
    :param n: Count of equal size split parts.
    :param ms: Minimum Support (Freqmine work size attribute).
    """

    prefixfolder = 'inputfiles'
    parsecconffolder = os.path.join(prefixfolder,'parsecconf')
    pkgconffolder = os.path.join(prefixfolder,'pkgconf')
    os.mkdir(prefixfolder)
    os.mkdir(parsecconffolder)
    os.mkdir(pkgconffolder)
    tarfilename = os.path.basename(tfile).split('.')[0]
    tar = tarfile.open(tfile)
    fm = tar.getmembers()
    if len(fm) != 1:
        print('Error: Tar File with more then one file member!')
    else:
        fm = fm[0]
    f = tar.extractfile(fm)
    numberoflines = sum(1 for line in f)
    splitlen = numberoflines // n
    print('Splitting input file %s on %s files' % (tfile,n))
    print('Original file total lines = ',numberoflines)
    print('Split length = ',splitlen)
    partscount = 1
    fd = None
    f = tar.extractfile(fm)
    lfile = []
    for line,linetxt in enumerate(f):
        if line == 0:
            newfilename = fm.name.split('.')[0] + '_' + ('%02d' % partscount) + '.' + fm.name.split('.')[1]
            fd = open(newfilename,'w')
        elif line%splitlen == 0 and line<splitlen*n:
            fd.close()
            newtarfilename = os.path.join(prefixfolder,tarfilename + '_' + ('%02d' % partscount) + '.tar')
            print(partscount,newtarfilename)
            tar2 = tarfile.open(newtarfilename,'w')
            tar2.add(newfilename)
            tar2.close()
            os.remove(newfilename)
            lfile.append((newtarfilename,newfilename))
            partscount += 1
            newfilename = fm.name.split('.')[0] + '_' + str(partscount) + '.' + fm.name.split('.')[1]
            fd = open(newfilename,'w')
        fd.write(linetxt.decode())
    newtarfilename = os.path.join(prefixfolder,tarfilename + '_' + ('%02d' % partscount) + '.tar')
    print(partscount, newtarfilename)
    tar2 = tarfile.open(newtarfilename, 'w')
    tar2.add(newfilename)
    tar2.close()
    os.remove(newfilename)
    tar.close()
    lfile.append((newtarfilename, newfilename))
    for ftn,fn in lfile:
        print(fn)
        fnbase = os.path.basename(ftn).split('.')[0]
        fnbase = '_'.join(fnbase.split('_')[1:])
        fpath1 = os.path.join(parsecconffolder,fnbase+'.runconf')
        fconf1 = open(fpath1,'w')
        fconf1.write(template_parsecconf % (fnbase+'.runconf',fnbase))
        fconf1.close()
        fpath2 = os.path.join(pkgconffolder,fnbase+'.runconf')
        fconf2 = open(fpath2,'w')
        fconf2.write(template_freqmine_inputconf % (fnbase+'.runconf',fn,ms))
        fconf2.close()
    print('Sucessful finished split operation. Total parts = ',partscount)

def freqmine_inputfilesplitprogressive(tfile, n, ms):
    """
    Split Freqmine Benchmark input tar file within 'n' aritmetic progressive size
    parts of new tar files with names like originalname_1 ... originalname_n.

    :param tfile: Freqmine input tar file name to split.
    :param n: Count of aritmetic progressive size split parts.
    :param ms: Minimum Support (Freqmine work size attribute).
    """

    prefixfolder = 'inputfiles'
    parsecconffolder = os.path.join(prefixfolder,'parsecconf')
    pkgconffolder = os.path.join(prefixfolder,'pkgconf')
    os.mkdir(prefixfolder)
    os.mkdir(parsecconffolder)
    os.mkdir(pkgconffolder)
    tarfilename = os.path.basename(tfile).split('.')[0]
    tar = tarfile.open(tfile)
    fm = tar.getmembers()
    if len(fm) != 1:
        print('Error: Tar File with more then one file member!')
    else:
        fm = fm[0]
    f = tar.extractfile(fm)
    numberoflines = sum(1 for line in f)
    splitlenbase = numberoflines // n
    splitlen = splitlenbase
    print('Splitting input file %s on %s files' % (tfile,n))
    print('Original file total lines = ',numberoflines)
    print('Split length = ',splitlen)
    partscount = 1
    fd = None
    lfile = []
    eof = False
    while not eof:
        f = tar.extractfile(fm)
        for line,linetxt in enumerate(f):
            if line == 0:
                newfilename = fm.name.split('.')[0] + '_' + ('%02d' % partscount) + '.' + fm.name.split('.')[1]
                fd = open(newfilename,'w')
            elif line%splitlen == 0 and line<=splitlenbase*(n-1):
                fd.close()
                newtarfilename = os.path.join(prefixfolder,tarfilename + '_' + ('%02d' % partscount) + '.tar')
                print(partscount,newtarfilename)
                print(" line: ", line," splitlen: ",splitlen)
                tar2 = tarfile.open(newtarfilename,'w')
                tar2.add(newfilename)
                tar2.close()
                os.remove(newfilename)
                lfile.append((newtarfilename,newfilename))
                partscount += 1
                splitlen = partscount*splitlenbase
                break
            fd.write(linetxt.decode())
        print("*** line: ", line, " splitlen: ", splitlen)
        if line >= numberoflines-1:
            fd.close()
            eof = True
    newtarfilename = os.path.join(prefixfolder,tarfilename + '_' + ('%02d' % partscount) + '.tar')
    print(partscount, newtarfilename)
    print(" line: ", line, " splitlen: ", splitlen)
    tar2 = tarfile.open(newtarfilename, 'w')
    tar2.add(newfilename)
    tar2.close()
    os.remove(newfilename)
    tar.close()
    lfile.append((newtarfilename, newfilename))
    for ftn,fn in lfile:
        print(fn)
        fnbase = os.path.basename(ftn).split('.')[0]
        fnbase = '_'.join(fnbase.split('_')[1:])
        fpath1 = os.path.join(parsecconffolder,fnbase+'.runconf')
        fconf1 = open(fpath1,'w')
        fconf1.write(template_parsecconf % (fnbase+'.runconf',fnbase))
        fconf1.close()
        fpath2 = os.path.join(pkgconffolder,fnbase+'.runconf')
        fconf2 = open(fpath2,'w')
        fconf2.write(template_freqmine_inputconf % (fnbase+'.runconf',fn,ms))
        fconf2.close()
    print('Sucessful finished split operation. Total parts = ',partscount)

def argsparsevalidation():
    """
    Validation of passed script arguments.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to parse a folder with parsec log files')
    parser.add_argument('foldername', help='Foldername with parsec log files.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from the linux shell script run.
    Process a folder with parsec log files and return Dataframes and data plot.

    """

    runfiles = []

    args = argsparsevalidation()

    if os.path.isdir(args.foldername):
        runfiles = runlogfilesfilter(args.foldername)
        print("\nProcessing folder: ", args.foldername)
        print(len(runfiles),"files")
    else:
        print("Error: Folder name not found.")
        exit(1)

    if(runfiles):
        print("\nTimes: \n")
        data = runlogfilesprocess(args.foldername,runfiles)
        print(data)
        print("\nSpeedups: \n")
        spdata = speedupframebuild(data)
        print(spdata)
        plot2Dspeedup(spdata)
        plot3Dspeedup(spdata)
    else:
        print("Warning: Folder is empty")
        exit(1)

if __name__ == '__main__':
    main()

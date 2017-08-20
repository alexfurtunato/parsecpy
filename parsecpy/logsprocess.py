#!python3
#  -*- coding: utf-8 -*-
"""
  Module to process parsec output informations.

"""

import os
import argparse
from datetime import datetime
import json

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

def runlogfilesprocess(fn, fl):
    """
    Process parsec log files within a folder.

    :param fn: Folder name.
    :param fl: Filename list of parsec log files.
    :return: dataframe with extracted data.
    """

    datadict = {}
    benchmarks = set()
    for filename in fl:
        fpath = os.path.join(fn, filename)
        fattrs = fileprocess(fpath)
        datadict = datadictbuild(datadict, fattrs)
        benchmarks.add(fattrs['benchmark'])
    return {'benchmarks': list(benchmarks), 'data': datadict}

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

def argsparsevalidation():
    """
    Validation of passed script arguments.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to parse a folder with parsec log files and save measures an output file')
    parser.add_argument('foldername', help='Foldername with parsec log files.')
    parser.add_argument('outputfilename', help='Filename to save the measures dictionary.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from the linux shell script run.
    Process a folder with parsec log files and return Dataframes.

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
        print("\nProcessed Data: \n")
        resultdata = runlogfilesprocess(args.foldername,runfiles)
        outputdict = {'config': {}, 'data': resultdata['data']}
        execdate = datetime.now()
        outputdict['config']['pkg'] = ', '.join(resultdata['benchmarks'])
        outputdict['config']['execdate'] = execdate.strftime("%d-%m-%Y_%H:%M:%S")
        outputdict['config']['command'] = 'logsprocess folder: '+ args.foldername
        fdatename = execdate.strftime("%Y-%m-%d_%H:%M:%S")
        with open('logs_processed_'+fdatename + args.outputfilename, 'w') as f:
            json.dump(outputdict, f, ensure_ascii=False)
        print(outputdict)
    else:
        print("Warning: Folder is empty")
        exit(1)

if __name__ == '__main__':
    main()
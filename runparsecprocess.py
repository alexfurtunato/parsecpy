#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import dataprocess
import shlex
from datetime import datetime
import argparse
import json


def argsparseintlist(txt):
    """
    Validate the list of cores argument.

    :param txt: argument of comma separated number of cores list.
    :return: list of integer converted number of cores.
    """

    txt = txt.split(',')
    listarg = [int(i) for i in txt]
    return listarg

def argsparseinputlist(txt):
    """
    Validate the single or multiple input names argument.
     - Formats:
       - Single: one input name string. Ex: native.
       - Multiple: input names with sequential range numbers. Ex: native2:5

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
                inputsets = [iname + ('%02d' % i) for i in range(iinit, ifinal + 1)]
            else:
                msg = "Wrong compost inputset name syntax: inputsetname[<initialnumber>:<finalnumber>]. Ex: native_1:10"
                raise argparse.ArgumentTypeError(msg)
        else:
            msg = "Wrong compost inputset name syntax: <inputsetname>[<initialnumber>:<finalnumber>]. Ex: native_1:10"
            raise argparse.ArgumentTypeError(msg)
    else:
        msg = "Wrong compost inputset name syntax: <inputsetname>[<initialnumber>:<finalnumber>]. Ex: native_1:10"
        raise argparse.ArgumentTypeError(msg)
    return inputsets


def argsparsevalidation():
    """
    Validation of passed script arguments.

    :return: argparse object with validated arguments.
    """

    choicebuilds=['gcc','gcc-serial','gcc-hooks','gcc-openmp','gcc-pthreads', 'gcc-tbb']
    helpinputtxt = 'Input name to be used on run. (Default: %(default)s). '
    helpinputtxt += 'Syntax: inputsetname[<initialnumber>:<finalnumber>]. Ex: native or native_1:10'
    parser = argparse.ArgumentParser(description='Script to split a input parsec file')
    parser.add_argument('c', type=argsparseintlist,help='List of cores numbers to be used. Ex: 1,2,4')
    parser.add_argument('-p','--package', help='Package Name to run', required=True)
    parser.add_argument('-c','--compiler', help='Compiler name to be used on run. (Default: %(default)s).', choices=choicebuilds, default='gcc-hooks')
    parser.add_argument('-i','--input', type=argsparseinputlist, help=helpinputtxt, default='native')
    parser.add_argument('-r','--repititions', type=int, help='Number of repititions for a specific run. (Default: %(default)s)', default=1)
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from the linux shell script run.
    Trigger a sequence of parsecmgmt programm runs and save the Dataframes of data within files.

    """

    datadict = {}
    command = 'parsecmgmt -a run -p %s -c %s -i %s -n %s'

    args = argsparsevalidation()
    datadict['config'] = {'pkg': args.package, 'command': command % (args.package, args.compiler,i, c)}
    datadict['data'] = {}
    print("Processing %s Repetitions: " % (args.repititions))
    for i in args.input:
        for c in args.c:
            print("- Inputset: ", i, "with %s cores" % c)
            for r in range(args.repititions):
                print("\n*** Execution ",r+1)
                try:
                    cmd = shlex.split(command % (args.package, args.compiler,i, c))
                    res = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    output, error = res.communicate()
                    if output:
                        attrs = dataprocess.contentextract(output.decode())
                        datadict['data'] = dataprocess.datadictbuild(datadict['data'], attrs, c)
                    if error:
                        print("Error: Execution return error code = ",res.returncode)
                        print("Error Message: ", error.strip())
                    print("\n### Saida ###")
                    print(output.decode())
                except OSError as e:
                    print("Error: Error from OS. Return Code = ",e.errno)
                    print("Error Message: ", e.strerror)
                except:
                    print("Error: Error on System Execution : ", sys.exc_info())

    dataname = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(args.package + '_datafile_' + dataname + '.dat', 'w') as f:
        json.dump(datadict,f,ensure_ascii=False)

#    dataf = dataprocess.dataframebuild(datadict)
#    dataf.to_json('timedatafile_'+dataname+'.dat')

    print("\n Data Dictionary: ")
    print(datadict)

#    print("\n Resume Dataframe: ")
#    print(dataf)

#    print("\n Resume Speedups Dataframe: ")
#    dfs = dataprocess.speedupframebuild(dataf)
#    dfs.to_json('speeddatafile_'+dataname+'.dat')
#    print(dfs)

if __name__ == '__main__':
    main()
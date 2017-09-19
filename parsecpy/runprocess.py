#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a parsec application.

    Its possible use a loop to repeat the same single parsec application
    run on specific number of times; And, also, its possible to refer
    differents input sizes to generate executions and resume all on a
    Pandas Dataframe with times and speedups.

    parsecpy_runprocess [-h] -p PACKAGE
        [-c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}]
        [-i INPUT] [-r REPITITIONS] c

    Script to run parsec app with repetitions and multiples inputs sizes

    positional arguments
        c
            List of cores numbers to be used. Ex: 1,2,4

    optional arguments
        -h, --help
            show this help message and exit
        -p PACKAGE, --package PACKAGE
            Package Name to run
        -c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb},
            --compiler {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}
            Compiler name to be used on run. (Default: gcc-hooks).
        -i INPUT, --input INPUT
            Input name to be used on run. (Default: native).
            Syntax: inputsetname[<initialnumber>:<finalnumber>]. Ex: native or native_1:10
        -r REPITITIONS, --repititions REPITITIONS
            Number of repititions for a specific run. (Default: 1)
    Example
        parsecpy_runprocess -p frqmine -c gcc-hooks -r 5 -i native 1,2,4,8
"""

import argparse
import shlex
import subprocess
import sys
import os
from datetime import datetime

from parsecpy.dataprocess import ParsecData

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


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    compilerchoicebuilds=['gcc','gcc-serial','gcc-hooks','gcc-openmp',
                          'gcc-pthreads', 'gcc-tbb']
    helpinputtxt = 'Input name to be used on run. (Default: %(default)s). ' \
                   'Syntax: inputsetname[<initialnumber>:<finalnumber>]. ' \
                   'Ex: native or native_1:10'
    parser = argparse.ArgumentParser(description='Script to run parsec app '
                                                 'with repetitions and '
                                                 'multiples inputs sizes')
    parser.add_argument('c', type=argsparseintlist,
                        help='List of cores numbers to be '
                             'used. Ex: 1,2,4')
    parser.add_argument('-p','--package', help='Package Name to run',
                        required=True)
    parser.add_argument('-c','--compiler',
                        help='Compiler name to be used on run. '
                             '(Default: %(default)s).',
                        choices=compilerchoicebuilds, default='gcc-hooks')
    parser.add_argument('-i','--input', type=argsparseinputlist,
                        help=helpinputtxt, default='native')
    parser.add_argument('-r','--repititions', type=int,
                        help='Number of repititions for a specific run. '
                             '(Default: %(default)s)', default=1)
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    command = 'parsecmgmt -a run -p %s -c %s -i %s -n %s'

    args = argsparsevalidation()
    rundate = datetime.now()
    hostname = os.uname()[1]
    datarun = ParsecData()
    datarun.config = {'pkg': args.package, 'execdate': rundate,
                   'command': command % (args.package, args.compiler,
                                         args.input, args.c),
                      'hostname': hostname}
    print("Processing %s Repetitions: " % (args.repititions))
    for i in args.input:
        for c in args.c:
            print("- Inputset: ", i, "with %s cores" % c)
            for r in range(args.repititions):
                print("\n*** Execution ",r+1)
                try:
                    cmd = shlex.split(command % (args.package,
                                                 args.compiler,i, c))
                    res = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT)
                    output, error = res.communicate()
                    if output:
                        attrs = datarun.contentextract(output.decode())
                        datarun.measurebuild(attrs, c)
                    if error:
                        print("Error: Execution return error code = ",
                              res.returncode)
                        print("Error Message: ", error.strip())
                    print("\n### Saida ###")
                    print(output.decode())
                except OSError as e:
                    print("Error: Error from OS. Return Code = ",e.errno)
                    print("Error Message: ", e.strerror)
                except:
                    print("Error: Error on System Execution : ", sys.exc_info())
    print("\n Resume: ")
    print(datarun)
    print(datarun.times())
    datarun.savedata()

if __name__ == '__main__':
    main()
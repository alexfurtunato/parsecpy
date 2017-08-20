#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Module to run a parsec application.

  Its possible run on loop to repeat the same sigle parsec execution on specific number of times;
  And, also, its possiblme refer to multiples input sizes on the same execution.

"""
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime

from logsprocess import contentextract,datadictbuild


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

    compilerchoicebuilds=['gcc','gcc-serial','gcc-hooks','gcc-openmp','gcc-pthreads', 'gcc-tbb']
    helpinputtxt = 'Input name to be used on run. (Default: %(default)s). '
    helpinputtxt += 'Syntax: inputsetname[<initialnumber>:<finalnumber>]. Ex: native or native_1:10'
    parser = argparse.ArgumentParser(description='Script to run parsec app with repetitions and multiples inputs sizes')
    parser.add_argument('c', type=argsparseintlist,help='List of cores numbers to be used. Ex: 1,2,4')
    parser.add_argument('-p','--package', help='Package Name to run', required=True)
    parser.add_argument('-c','--compiler', help='Compiler name to be used on run. (Default: %(default)s).', choices=compilerchoicebuilds, default='gcc-hooks')
    parser.add_argument('-i','--input', type=argsparseinputlist, help=helpinputtxt, default='native')
    parser.add_argument('-r','--repititions', type=int, help='Number of repititions for a specific run. (Default: %(default)s)', default=1)
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from the linux shell script run.
    Trigger a sequence of parsecmgmt program runs and save the Dataframes of data within files.

    """

    datadict = {}
    command = 'parsecmgmt -a run -p %s -c %s -i %s -n %s'

    args = argsparsevalidation()
    datadict['config'] = {'pkg': args.package, 'command': command % (args.package, args.compiler,args.input, args.c)}
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
                        attrs = contentextract(output.decode())
                        datadict['data'] = datadictbuild(datadict['data'], attrs, c)
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

    execdate = datetime.now()
    datadict['config']['execdate'] = execdate.strftime("%d-%m-%Y_%H:%M:%S")
    fdatename = execdate.strftime("%Y-%m-%d_%H:%M:%S")
    with open(args.package + '_datafile_' + fdatename + '.dat', 'w') as f:
        json.dump(datadict,f,ensure_ascii=False)

    print("\n Data Dictionary: ")
    print(datadict)

if __name__ == '__main__':
    main()
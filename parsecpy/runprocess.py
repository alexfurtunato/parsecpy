#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a parsec application.

    Its possible use a loop to repeat the same single parsec application
    run on specific number of times; And, also, its possible to refer
    differents input sizes to generate executions and resume all on a
    DataArray (xArray module) with times, speedups or efficiency.

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
            --compiler {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,
            gcc-tbb}. Compiler name to be used on run. (Default: gcc-hooks).
        -f FREQUENCY, --frequency FREQUENCY
            List of frequencies (KHz). Ex: 2000000, 2100000
        -i INPUT, --input INPUT
            Input name to be used on run. (Default: native).
            Syntax: inputsetname[<initialnumber>:<finalnumber>].
            Ex: native or native_1:10
        -r REPITITIONS, --repetitions REPITITIONS
            Number of repetitions for a specific run. (Default: 1)
        -b CPU_BASE, --cpubase CPU_BASE
            If run with thread affinity(limiting the running cores to defined
            number of cores), define the cpu base number.
        -v VERBOSITY, --verbosity VERBOSITY
            verbosity level. 0 = No verbose

    Example
        parsecpy_runprocess -p frqmine -c gcc-hooks -r 5 -i native 1,2,4,8
"""

import argparse
import shlex
import subprocess
import sys
import os
import time
from datetime import datetime
from cpufreq import cpuFreq,CPUFreqErrorInit

from parsecpy.dataprocess import ParsecData
from parsecpy import argsparseintlist, argsparseinputlist, procs_list


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    compilerchoicebuilds = ['gcc', 'gcc-serial', 'gcc-hooks', 'gcc-openmp',
                            'gcc-pthreads', 'gcc-tbb']
    helpinputtxt = 'Input name to be used on run. (Default: %(default)s). ' \
                   'Syntax: inputsetname[<initialnumber>:<finalnumber>]. ' \
                   'From lowest to highest size. Ex: native or native_1:10'
    parser = argparse.ArgumentParser(description='Script to run parsec app '
                                                 'with repetitions and '
                                                 'multiples inputs sizes')
    parser.add_argument('c', type=argsparseintlist,
                        help='List of cores numbers to be '
                             'used. Ex: 1,2,4')
    parser.add_argument('-p', '--package', help='Package Name to run',
                        required=True)
    parser.add_argument('-c', '--compiler',
                        help='Compiler name to be used on run. '
                             '(Default: %(default)s).',
                        choices=compilerchoicebuilds, default='gcc-hooks')
    parser.add_argument('-f', '--frequency', type=argsparseintlist,
                        help='List of frequencies (KHz). Ex: 2000000,2100000')
    parser.add_argument('-i', '--input', type=argsparseinputlist,
                        help=helpinputtxt, default='native')
    parser.add_argument('-r', '--repetitions', type=int,
                        help='Number of repetitions for a specific run. '
                             '(Default: %(default)s)', default=1)
    parser.add_argument('-b', '--cpubase', type=int,
                        help='If run with thread affinity(limiting the '
                             'running cores to defined number of cores), '
                             'define the cpu base number.')
    parser.add_argument("--threadmon", type=bool,
                        help="Enable thread monitoring", default=False)
    parser.add_argument("--ipmi", help="Enable ipmi measuments",
                        nargs=3, metavar=("server", "user", "password"))
    parser.add_argument('-v', '--verbosity', type=int,
                        help='verbosity level. 0 = No verbose', default=0)
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    command = 'parsecmgmt -a run -p %s -c %s -i %s -n %s'

    args = argsparsevalidation()
    rundate = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    hostname = os.uname()[1]
    datarun = ParsecData()
    datarun.config = {'pkg': args.package,
                      'execdate': rundate,
                      'command': ' '.join(sys.argv),
                      'input_sizes': args.input,
                      'thread_cpu': {},
                      'hostname': hostname}
    if args.frequency:
        try:
            cf = cpuFreq()
            freq_avail = cf.available_frequencies
            if not set(args.frequency).issubset(set(freq_avail)):
                print("ERROR: Available CPUs frequencies aren't compatibles "
                      "with frequency list passed on execution frequeny "
                      "argument (--frequency)")
                exit(1)
            else:
                cf.set_governors("userspace")
                freqs = args.frequency
                print("Running with governor 'userspace' and frequencies %s.\n"
                      % str(freqs))
        except CPUFreqErrorInit as err:
            print(err)
        except:
            print("ERROR: Unknown error on frequencies list.", sys.exc_info()[0])
            print(sys.exc_info())
            sys.exit(1)
    else:
        freqs = [0]
        try:
            cf = cpuFreq()
            cf.set_governors("ondemand")
            print("Running with governor 'ondemand'.\n")
        except:
            print("WARNING: Governor set is not possible.")

    if args.cpubase:
        env = os.environ
        env['PARSEC_CPU_BASE'] = str(args.cpubase)

    sensor = []
    if args.ipmi:
        from ipmi import IPMI
        ipmi_sensor = IPMI(
            server=args.ipmi[0], user=args.ipmi[1], password=args.ipmi[2])
        sensor.append(ipmi_sensor)

    print("Processing %s Repetitions: " % args.repetitions)
    for f in freqs:
        ftxt = None
        if not (len(freqs) == 1 and f == 0):
            try:
                cf.set_frequencies(f)
                ftxt = "Frequency: %s," % f
            except:
                print("ERROR: Unknown error on frequencies setting.")
                print(sys.exc_info())
                sys.exit(1)
        for i,inputsize in enumerate(args.input):
            for c in args.c:
                print("\n- %s Inputset: %s with %s cores"
                      % (ftxt, inputsize, c))
                for r in range(args.repetitions):
                    print("\n*** Execution ", r+1)
                    try:
                        if args.cpubase:
                            env['PARSEC_CPU_NUM'] = str(c)
                        cmd = shlex.split(command % (args.package,
                                                     args.compiler,
                                                     inputsize, c))
                        res = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT)
                        procs = None
                        ti = time.perf_counter()
                        while res.poll() is None:
                            if args.threadmon:
                                try:
                                    procs = procs_list(args.package, procs)
                                except:
                                    print("ERROR: get thread processors list")
                                    continue
                            if sensor:
                                for sen in sensor:
                                    try:
                                        sensor_data = sen.get_data()
                                        datarun.powerbuild(
                                            attrs={
                                                "ipmi": sensor_data,
                                                "time": time.perf_counter() - ti
                                            },
                                            keys=[f, i+1, c, r+1]
                                        )
                                        if args.verbose > 2:
                                            print("Sensor data", sensor_data)
                                        time.sleep(0.1)
                                    except Exception as e:
                                        print(e)
                                        print("ERROR: get sensor data")
                                        continue
                        if args.verbosity > 1 and args.threadmon:
                            print('\nCPUs Id per Thread:')
                            print(procs)
                            print('\n')
                        if res.returncode != 0:
                            error = res.stdout.read()
                            print('Error Code: ', res.returncode)
                            print('Error Message: ', error.decode())
                        else:
                            if args.threadmon:
                                datarun.threadcpubuild(procs, f, inputsize, c)
                            output = res.stdout.read()
                            if output:
                                if args.verbosity > 2:
                                    print('\nParsec Output:')
                                    print(output.decode())
                                    print('\n')
                                attrs = datarun.contentextract(output.decode())
                                datarun.measurebuild(attrs=attrs,
                                                     frequency=f,
                                                     inputsize = i+1,
                                                     numberofcores=c)
                    except OSError as e:
                        print("Error: Error from OS. Return Code = ", e.errno)
                        print("Error Message: ", e.strerror)
                    except:
                        print("Error: Error on System Execution : ",
                              sys.exc_info())
    print(datarun)
    print(datarun.times())
    print("\n\n***** Done! *****\n")
    if args.frequency:
        try:
            cf.reset()
            print("Resetting governors to 'ondemand'.")
        except:
            print("ERROR: resetting governors.")
            print(sys.exc_info())
            sys.exit(1)
    fn = datarun.savedata()
    print('Running data saved on filename: %s' % fn)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Module to run a model of a parsec application.

    Its possible define the number of threads to execute a model
    on a fast way; The modelfunc to represent the application should be
    provided by user on a python module file. Its possible, also, provide a
    overhead function to integrate the model

    usage: parsecpy_runmodel_pso [-h] -f PARSECPYFILENAME -l LOWERVALUES -u
                                 UPPERVALUES [-o OVERHEAD] [-x MAXITERATIONS]
                                 [-p PARTICLES] [-t THREADS] [-r REPETITIONS] -m
                                 MODELFILEABSPATH

    Script to run swarm modelling to predict aparsec application output

    optional arguments:
      -h, --help            show this help message and exit
      -f PARSECPYFILENAME, --parsecpyfilename PARSECPYFILENAME
                            Input filename from Parsec specificated package.
      -l LOWERVALUES, --lowervalues LOWERVALUES
                            List of minimum particles values used. Ex: -1,0,-2,0
      -u UPPERVALUES, --uppervalues UPPERVALUES
                            List of maximum particles values used. Ex: 5,2,1,10
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -x MAXITERATIONS, --maxiterations MAXITERATIONS
                            Number max of iterations
      -p PARTICLES, --particles PARTICLES
                            Number of particles
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to algorithm execution
      -m MODELFILEABSPATH, --modelfileabspath MODELFILEABSPATH
                            Absolute path from Python file with theobjective
                            function.
    Example
        parsecpy_runmodel_pso -l -10,-10,-10,-10,-10 -u 10,10,10,10,10
            -f /var/myparsecsim.dat -m /var/mymodelfunc.py -x 1000 -p 10
"""

import os
import sys
import time
import argparse
from parsecpy import ParsecData
from parsecpy import Swarm

def argsparsefloatlist(txt):
    """
    Validate the list int argument.

    :param txt: argument of comma separated int strings.
    :return: list of integer converted ints.
    """

    txt = txt.split(',')
    listarg = [float(i.strip()) for i in txt]
    return listarg

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to run swarm '
                                                 'modelling to predict a'
                                                 'parsec application output')
    parser.add_argument('-f','--parsecpyfilename', required=True,
                        help='Input filename from Parsec specificated package.')
    parser.add_argument('-l', '--lowervalues', type=argsparsefloatlist, required=True,
                        help='List of minimum particles values '
                             'used. Ex: -1,0,-2,0')
    parser.add_argument('-u', '--uppervalues', type=argsparsefloatlist, required=True,
                        help='List of maximum particles values '
                             'used. Ex: 5,2,1,10')
    parser.add_argument('-o','--overhead', type=bool,
                        help='If it consider the overhead', default=False)
    parser.add_argument('-x','--maxiterations', type=int,
                        help='Number max of iterations', default=100)
    parser.add_argument('-p','--particles', type=int,
                        help='Number of particles', default=100)
    parser.add_argument('-t','--threads', type=int,
                        help='Number of Threads', default=1)
    parser.add_argument('-r','--repetitions', type=int,
                        help='Number of repetitions to algorithm execution', default=10)
    parser.add_argument('-m','--modelfileabspath', required=True,
                        help='Absolute path from Python file with the'
                             'objective function.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    print("Processing the Model")

    # adjust list of arguments to avoid negative number values error
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg

    args = argsparsevalidation()

    if os.path.isfile(args.modelfileabspath):
        modelpath = args.modelfileabspath
    else:
        print('Error: You should inform the correct module of objective function to model')

    l = args.lowervalues
    u = args.uppervalues

    parsec_exec = ParsecData(args.parsecpyfilename)
    y_measure = parsec_exec.speedups()
    N = [(col, int(col.split('_')[1])) for col in y_measure]
    p = y_measure.index
    argsswarm = (y_measure, args.overhead, p, N)


    repetitions = range(args.repetitions)
    err_min = 0
    computed_models = []
    best_model_idx = 0

    starttime = time.time()
    for i in repetitions:
        print('Algorithm Execution: ',i+1, '\n')

        S = Swarm(l, u, args=argsswarm, threads=args.threads,
                  size=args.particles, w=1, c1=1, c2=4,
                  maxiter=args.maxiterations, modelpath=modelpath)
        model = S.run()
        computed_models.append(model)
        if i == 0:
            err_min = model.error
            print('Error: ', err_min)
        else:
            if model.error < err_min:
                best_model_idx = i
                print('Error: ', err_min, '->', model.error)
                err_min = model.error
        endtime = time.time()
        print('Execution time = %s seconds' % (endtime - starttime))
        starttime = endtime

    print('\n\n***** Done! *****\n')
    print('Error: %.8f \nPercentual Error (Measured Mean): %.8f %%' %
          (computed_models[best_model_idx].error,
           computed_models[best_model_idx].errorrel))
    print('Best Parameters: \n',computed_models[best_model_idx].params)
    print('\nMeasured Speedup: \n',y_measure)
    print('\nModeled Speedup: \n',computed_models[best_model_idx].y_model)

    fn = computed_models[best_model_idx].savedata(parsec_exec.config)
    print('Model data saved on filename: %s' % fn)

if __name__ == '__main__':
    main()
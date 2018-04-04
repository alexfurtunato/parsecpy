parsecpy
========

Python library to interface with PARSEC 2.1 and 3.0 Benchmark,
controlling execution triggers and processing the output measures times
data to calculate speedups. Further, the library can generate a
mathematical model of speedup of a parallel application, based on
"Particles Swarm Optimization" or "Coupled Simulating Annealing"
algorithms to discover the parameters that minimize a "objective
function". The objective function can be build with a module python
passed as argument to library on execution script.

Features
--------

-  Run parsec application with multiple input sizes and, optionally,
   repet the execution to better find outs.
-  Process a group of Parsec 2.1 or 3.0 logs files, generates from a
   shell direct execution of parsec.
-  Manipulate of resulting data from logs process or online execution,
   obtained by module run script itself.
-  Calculate the speedups and efficency of applications, if it's
   possible, using the measured times of execution.
-  provide a "PSO" algorithm to model the speedup of a parallel
   application with regression process.
-  Provide a "CSA" algorithm to model the speedup of a parallel
   application with regression process.
-  Calculate statistics scores of model data using cross validate
   process.

Prerequisites
-------------

-  Parsec 2.1 or newer
-  Python3 or newer
-  Numpy
-  Pandas
-  Matplotlib with Mplot3D Toolkit (Optional, to plot 3D surface)
-  Scikit-learn

Site
----

-  https://github.com/alexfurtunatoifrn/parsecpy

Installation
------------

::

    $ pip3 install parsecpy

Usage
-----

Class ParsecData
~~~~~~~~~~~~~~~~

::

    >>> from parsecpy import ParsecData
    >>> d = ParsecData('path_to_datafile')
    >>> print(d)        # Print summary informations
    >>> d.times()       # Show a Dataframe with mesures times
    >>> d.speedups()    # Show a Dataframe with speedups
    >>> d.plot3D(d.speedups(), title='Speedup', zlabel='speedup')   # plot a 3D Plot : speedups x number of cores x input sizes
    >>> d.plot3D(d.efficiency(), title='Efficiency', zlabel='efficiency')   # plot a 3D Plot : speedups x number of cores x input sizes

Class ParsecLogsData
~~~~~~~~~~~~~~~~~~~~

::

    >>> from parsecpy import ParsecLogsData
    >>> l = ParsecLogsData('path_to_folder_with_logfiles')
    >>> print(l)        # Print summary informations
    >>> l.times()       # Show a Dataframe with mesures times
    >>> l.speedups()    # Show a Dataframe with speedups
    >>> l.plot3D()      # plot a 3D Plot : speedups x number of cores x input sizes

Class Swarm
~~~~~~~~~~~

::

    >>> from parsecpy import Swarm
    >>> parsec_date = ParsecData("my_output_parsec_file.dat")
    >>> out_measure = parsec_exec.speedups()
    >>> inputsizes = [(col, int(col.split('_')[1])) for col in y_measure]
    >>> cores = y_measure.index
    >>> overhead = False
    >>> argsswarm = (out_measure, overhead, cores, inputsizes)
    >>> pso = Swarm([0,0,0,0], [2.0,1.0,1.0,2.0], args=argsswarm, threads=10, 
                    size=100, maxiter=1000, modelpath=/root/mymodelfunc.py)
    >>> model = pso.run()
    >>> model.validate()
    >>> print(model.params)
    >>> print(model.validation)

Class CoupledAnnealer
~~~~~~~~~~~~~~~~~~~~~

::

    >>> from parsecpy import CoupledAnnealer
    >>> parsec_date = ParsecData("my_output_parsec_file.dat")
    >>> out_measure = parsec_exec.speedups()
    >>> inputsizes = [(col, int(col.split('_')[1])) for col in y_measure]
    >>> cores = y_measure.index
    >>> overhead = False
    >>> argscsa = (out_measure, overhead, cores, inputsizes)
    >>> initial_state = [
            tuple((random.normalvariate(0, 5) for _ in xrange(args.dimension)))
            for x in xrange(args.annealers)]
    >>> csa = CoupledAnnealer(n_annealers=10, initial_state=initial_state, tgen_initial=0.01, tacc_initial=0.1,
                    threads=10, steps=1000, update_interval=100, dimension=5, 
                    args=argscsa, modelpath=/root/mymodelfunc.py)
    >>> model = csa.run()
    >>> print(model.params)

Requirements for model python module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The python module file provided by user should has the following
requirements:

-  Should has, at least, two function as following:

   ::

       def constraint_function(p, *args):
           # your code
           # arguments: 
           # p - particle object
           # args - list of position arguments passed to function:
           #   args[0] - Pandas Dataframe object of measured speedups (PasecData speedups)     
           #   args[1] - boolean value (if overhead should be considerable)
           #   args[2] - list of number of cores used on args[0] measured speedups
           #   args[3] - list of number of problems sizes used on args[0] measured speedups
           # analize the feasable of particles position (searched parameters)
           # return True or False, depend of requirements
           return boolean_value

       def objective_function(p, *args):
           # your code
           # calculate the function with should be minimized
           # return the calculated value
           return float_value 

Run Parsec
~~~~~~~~~~

Script to run parsec app with repetitions and multiples inputs sizes

::

    parsecpy_runprocess [-h] -p PACKAGE
                           [-c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}]
                           [-i INPUT] [-r REPITITIONS]
                           c

    positional arguments:
      c                     List of cores numbers to be used. Ex: 1,2,4

    optional arguments:
      -h, --help            show this help message and exit
      -p PACKAGE, --package PACKAGE
                            Package Name to run
      -c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}, --compiler {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}
                            Compiler name to be used on run. (Default: gcc-hooks).
      -i INPUT, --input INPUT
                            Input name to be used on run. (Default: native).
                            Syntax: inputsetname[<initialnumber>:<finalnumber>].
                            Ex: native or native_1:10
      -r REPITITIONS, --repititions REPITITIONS
                            Number of repititions for a specific run. (Default: 1)
                            
    Example:
        parsecpy_runprocess -p frqmine -c gcc-hooks -r 5 -i native 1,2,4,8

Run PSO Modelling script
~~~~~~~~~~~~~~~~~~~~~~~~

Script to run swarm modelling to predict aparsec application output

::

    parsecpy_runmodel_pso [-h] -f PARSECPYFILENAME -l LOWERVALUES -u
                                 UPPERVALUES [-o OVERHEAD] [-x MAXITERATIONS]
                                 [-p PARTICLES] [-t THREADS] [-r REPETITIONS] -m
                                 MODELFILEABSPATH

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

Run CSA Modelling script
~~~~~~~~~~~~~~~~~~~~~~~~

Script to run csa modelling to predict aparsec application output

::

    parsecpy_runmodel_csa [-h] -f PARSECPYFILENAME [-o OVERHEAD]
                                 [-d DIMENSION] [-s STEPS] [-u UPDATE_INTERVAL]
                                 [-a ANNEALERS] [-t THREADS] [-r REPETITIONS] -m
                                 MODELFILEABSPATH [-v VERBOSE]

    optional arguments:
      -h, --help            show this help message and exit
      -f PARSECPYFILENAME, --parsecpyfilename PARSECPYFILENAME
                            Input filename from Parsec specificated package.
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -d DIMENSION, --dimension DIMENSION
                            Number of parameters
      -s STEPS, --steps STEPS
                            Number max of iterations
      -u UPDATE_INTERVAL, --update_interval UPDATE_INTERVAL
                            Number steps to run cooling temperatures
      -a ANNEALERS, --annealers ANNEALERS
                            Number of annealers
      -t THREADS, --threads THREADS
                            Number of Threads
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions to algorithm execution
      -m MODELFILEABSPATH, --modelfileabspath MODELFILEABSPATH
                            Absolute path from Python file with theobjective
                            function.
      -v VERBOSE, --verbose VERBOSE
                            If it shows output verbosily: Values: 0, 1, 2

    Example
        parsecpy_runmodel_csa -f /var/myparsecsim.dat -m /var/mymodelfunc.py -d 5 -s 1000 -a 10

Logs process
~~~~~~~~~~~~

Script to parse a folder with parsec log files and save measures an
output file

::

    parsecpy_processlogs [-h] foldername outputfilename

    positional arguments:
      foldername      Foldername with parsec log files.
      outputfilename  Filename to save the measures dictionary.

    optional arguments:
      -h, --help      show this help message and exit

    Example:
        parsecpy_processlogs logs_folder my-logs-folder-data.dat

Create split parts
~~~~~~~~~~~~~~~~~~

Script to split a parsec input file on specific parts

::

    parsecpy_createinputs [-h] -p {freqmine,fluidanimate} -n NUMBEROFPARTS
                               [-t {equal,diff}] -x EXTRAARG
                               inputfilename

    positional arguments:
      inputfilename         Input filename from Parsec specificated package.

    optional arguments:
      -h, --help            show this help message and exit
      -p {freqmine,fluidanimate}, --package {freqmine,fluidanimate}
                            Package name to be used on split.
      -n NUMBEROFPARTS, --numberofparts NUMBEROFPARTS
                            Number of split parts
      -t {equal,diff}, --typeofsplit {equal,diff}
                            Split on equal or diferent size partes parts
      -x EXTRAARG, --extraarg EXTRAARG
                            Specific argument: Freqmine=minimum support (11000),
                            Fluidanimate=Max number of frames

    Example:
        parsec_createinputs -p fluidanimate -n 10 -t diff -x 500 fluidanimate_native.tar

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

Class used to generate the measured times structure, to save such data
in a "json" file, to load a previously saved json data file, to
calculate the speedup or efficiency of application and to plot 2D or 3D
graph of time, speedup or efficiency versus the number of cores and
frequency or input size.

::

    >>> from parsecpy import ParsecData
    >>> d = ParsecData('path_to_datafile')
    >>> print(d)        # Print summary informations
    >>> d.times()       # Show a Dataframe with mesures times
    >>> d.speedups()    # Show a Dataframe with speedups
    >>> d.plot3D(d.speedups(), title='Speedup', zlabel='speedup')   # plot a 3D Plot : speedups x number of cores x input sizes
    >>> d.plot3D(d.efficiency(), title='Efficiency', zlabel='efficiency')   # plot a 3D Plot : speedups x number of cores x input sizes

Class ParsecModel
~~~~~~~~~~~~~~~~~

::

    Class used to generate the result of modeling of the application, using any of supported algorithms (PSO, CSA or
    SVR). The class allows to save the modeling results, load previously saved model data, and plot the model data
    together with the real measurements.

    >>> from parsecpy import ParsecModel
    >>> m = ParsecModel('path_to_model_datafile')
    >>> print(m)        # Print summary informations
    >>> print(m.measure)       # Show a Dataframe with mesures speedups
    >>> print(m.y_model)       # Show a Dataframe with modeled speedups
    >>> print(m.error)         # Show the Mean Squared Error between measured and modeled speedup
    >>> m.plot3D(title='Speedup', showmeasures=True)   # plot a 3D Plot with measurements and model data

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

    >>> from parsecpy import data_detach, Swarm, ParsecModel
    >>> parsec_date = ParsecData("my_output_parsec_file.dat")
    >>> out_measure = parsec_exec.speedups()
    >>> meas = data_detach(out_measure)
    >>> overhead = False
    >>> kwargsmodel = {'overhead':  overhead}
    >>> sw = Swarm([0,0,0,0], [2.0,1.0,1.0,2.0], kwargs=kwargsmodel, threads=10,
                    size=100, maxiter=1000, modelpath=/root/mymodelfunc.py,
                    x_meas=meas['x'], y_meas=meas['y'])
    >>> error, solution = sw.run()
    >>> model = ParsecModel(bsol=solution,
                            berr=error,
                            ymeas=out_measure,
                            modelcodesource=sw.modelcodesource,
                            modelexecparams=sw.get_parameters())
    >>> scores = model.validate(kfolds=10)
    >>> print(model.sol)
    >>> print(model.scores)

Class CoupledAnnealer
~~~~~~~~~~~~~~~~~~~~~

::

    >>> import numpy as np
    >>> import random
    >>> from parsecpy import data_detach, Swarm, ParsecModel
    >>> parsec_date = ParsecData("my_output_parsec_file.dat")
    >>> out_measure = parsec_exec.speedups()
    >>> meas = data_detach(out_measure)
    >>> overhead = False
    >>> kwargsmodel = {'overhead':  overhead}
    >>> initial_state = initial_state = np.array([np.random.uniform(size=5)
                                      for _ in range(10)])
    >>> csa = CoupledAnnealer(n_annealers=10, initial_state=initial_state,
                    tgen_initial=0.01, tacc_initial=0.1,
                    threads=10, steps=1000, update_interval=100, dimension=5,
                    args=argscsa, modelpath=/root/mymodelfunc.py
                    x_meas=meas['x'], y_meas=meas['y'])
    >>> error, solution = csa.run()
    >>> model = ParsecModel(bsol=solution,
                            berr=error,
                            measure=out_measure,
                            modelcodesource=csa.modelcodesource,
                            modelexecparams=csa.get_parameters())
    >>> scores = model.validate(kfolds=10)
    >>> print(model.sol)
    >>> print(model.scores)

Requirements for model python module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The python module file provided by user should has the following
requirements:

-  To PSO model, should has the constraint function as following:

   def constraint\_function(par, x\_meas, \*\*kwargs): # your code #
   arguments: # par - particle object # kwargs - Dict with extra
   parameters: # kwargs['overhead'] - boolean value (if overhead should
   be considerable) # analize the feasable of particles position
   (searched parameters) # return True or False, depend of requirements
   return boolean\_value

-  To CSA model, should has probe function as following:

   def probe\_function(par, tgen): # your code # arguments: # par -
   actual parameters values # tgen - actual temperature of generation #
   generate a new probe solution # return a list os parameters of probe
   solution return probe\_solution

-  And the models files should has a objective function as following:

   ::

       def objective_function(par, x_meas, y_meas, **kwargs):
           # your code
           # arguments:
           # par - particle object
           # x_meas - Measures array of independent variables
           # y_meas - Measures array of dependent variable
           # kwargs - Dict with extra parameters:
           #   kwargs['overhead'] - boolean value (if overhead should be considerable)
           # calculate the function with should be minimized
           # return the calculated value
           return float_value 

Run Parsec
~~~~~~~~~~

Script to run parsec app with repetitions and multiples inputs sizes

::

    usage: parsecpy_runprocess [-h] -p PACKAGE
                           [-c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}]
                           [-f FREQUENCY] [-i INPUT] [-r REPETITIONS]
                           [-b CPUBASE] [-v VERBOSITY]
                           c

    Script to run parsec app with repetitions and multiples inputs sizes

    positional arguments:
      c                     List of cores numbers to be used. Ex: 1,2,4

    optional arguments:
      -h, --help            show this help message and exit
      -p PACKAGE, --package PACKAGE
                            Package Name to run
      -c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}, --compiler {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}
                            Compiler name to be used on run. (Default: gcc-hooks).
      -f FREQUENCY, --frequency FREQUENCY
                            List of frequencies (KHz). Ex: 2000000, 2100000
      -i INPUT, --input INPUT
                            Input name to be used on run. (Default: native).
                            Syntax: inputsetname[<initialnumber>:<finalnumber>].
                            From lowest to highest size. Ex: native or native_1:10
      -r REPETITIONS, --repetitions REPETITIONS
                            Number of repetitions for a specific run. (Default: 1)
      -b CPUBASE, --cpubase CPUBASE
                            If run with thread affinity(limiting the running cores
                            to defined number of cores), define the cpu base
                            number.
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose

    Example:
        parsecpy_runprocess -p freqmine -c gcc-hooks -r 5 -i native 1,2,4,8 -v 3

Run PSO or CSA Modelling script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script to run swarm modelling to predict a parsec application output. On
examples folder, exists a template file of configurations parameters to
use on execution of this script

::

    usage: parsecpy_runmodel [-h] --config CONFIG -f PARSECPYFILEPATH
                             [-p PARTICLES] [-x MAXITERATIONS]
                             [-l LOWERVALUES] [-u UPPERVALUES]
                             [-n PROBLEMSIZES] [-o OVERHEAD] [-t THREADS]
                             [-r REPETITIONS] [-c CROSSVALIDATION]
                             [-v VERBOSITY]

    Script to run modelling algorithm to predict a parsec application output

    optional arguments:
      -h, --help            show this help message and exit
      --config CONFIG       Filepath from Configuration file configurations
                            parameters
      -p PARSECPYDATAFILEPATH, --parsecpydatafilepath PARSECPYDATAFILEPATH
                            Path from input data file from Parsec specificated
                            package.
      -f FREQUENCIES, --frequency FREQUENCIES
                            List of frequencies (KHz). Ex: 2000000, 2100000
      -n PROBLEMSIZES, --problemsizes PROBLEMSIZES
                            List of problem sizes to model used. Ex:
                            native_01,native_05,native_08
      -o OVERHEAD, --overhead OVERHEAD
                            If it consider the overhead
      -t THREADS, --threads THREADS
                            Number of Threads
      -c CROSSVALIDATION, --crossvalidation CROSSVALIDATION
                            If run the cross validation of modelling
      -m MEASURESFRACTION, --measuresfraction MEASURESFRACTION
                            Fraction of measures data to calculate the model
      -v VERBOSITY, --verbosity VERBOSITY
                            verbosity level. 0 = No verbose
    Example
        parsecpy_runmodel --config my_config.json
                          -p /var/myparsecsim.dat -c True -v 3

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

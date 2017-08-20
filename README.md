# parsecpy
python package to interface with PARSEC Benchmark

## Features

 - Run parsec application with repetitions e multiple input sizes and output data to file
 - Process a group of Parsec logs files generates from a shell direct execution of parsec
 - Manipulation of data resulting for logs process or execution obtained by package run script
 
## Usage

### Class ParsecData

    >>> from parsecpy import ParsecData
    >>>  d = ParsecData('path_to_datafile')
    >>> d.times()       # Show a Dataframe with mesures times
    >>> d.speedups()    # Show a Dataframe with speedups
    >>> d.plot3D()      # plot a 3D Plot : speedups x number of cores x input sizes

### Run Parsec

    runparsecprocess.py [-h] -p PACKAGE
                           [-c {gcc,gcc-serial,gcc-hooks,gcc-openmp,gcc-pthreads,gcc-tbb}]
                           [-i INPUT] [-r REPITITIONS]
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
      -i INPUT, --input INPUT
                            Input name to be used on run. (Default: native).
                            Syntax: inputsetname[<initialnumber>:<finalnumber>].
                            Ex: native or native_1:10
      -r REPITITIONS, --repititions REPITITIONS
                            Number of repititions for a specific run. (Default: 1)
    
### Logs process

    logsprocess.py [-h] foldername outputfilename
    
    Script to parse a folder with parsec log files and save measures an output
    file
    
    positional arguments:
      foldername      Foldername with parsec log files.
      outputfilename  Filename to save the measures dictionary.
    
    optional arguments:
      -h, --help      show this help message and exit


### Create split parts

    createinputsizes.py [-h] -p {freqmine,fluidanimate} -n NUMBEROFPARTS
                               [-t {equal,diff}] -x EXTRAARG
                               inputfilename
    
    Script to split a parsec input file on specific parts
    
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
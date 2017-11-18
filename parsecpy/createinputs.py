#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
    Module to create new input sizes on parsec packages.

    Actually it can create onle fluidanimate and fremine new inputs


    parsecpy_createinputs [-h] -p {freqmine,fluidanimate,dedup, x264} -n NUMBEROFPARTS
        [-t {equal,diff}] -x EXTRAARG inputfilename

        Script to split a parsec input file on specific parts

        positional arguments
            inputfilename
                Input filename from Parsec specificated package.

        optional arguments
            -h, --help
                show this help message and exit
            -p {freqmine,fluidanimate,dedup,x264}, --package {freqmine,fluidanimate, x264}
                Package name to be used on split.
            -n NUMBEROFPARTS, --numberofparts NUMBEROFPARTS
                Number of split parts
            -t {equal,diff}, --typeofsplit {equal,diff}
                Split on equal or diferent size partes parts
            -x EXTRAARG, --extraarg EXTRAARG
                Specific argument: Freqmine=minimum support (11000),
                Fluidanimate=Max number of frames
        Example
            parsec_createinputs -p fluidanimate -n 10 -t diff -x 500 fluidanimate_native.tar
"""

import os
import shutil
import argparse

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

# Template with 2 arguments: filename.runconf and filename.dat
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

# Template with 2 argument: filename.runconf, filename.iso
template_dedup_inputconf =  '''#!/bin/bash
# %s - file containing information necessary to run a specific
#                    program of the PARSEC benchmark suite

# Binary file to execute, relative to installation root
run_exec="bin/dedup"

# Arguments to use
run_args="-c -p -v -t ${NTHREADS} -i %s -o output.dat.ddp"
'''

# Template with 2 argument: filename.runconf, filename.y4m
template_x264_inputconf =  '''#!/bin/bash
# %s - file containing information necessary to run a specific
#                    program of the PARSEC benchmark suite

# Binary file to execute, relative to installation root
run_exec="bin/x264"

# Arguments to use
run_args="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto 
--b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 
--analyse b8x8,i4x4 --threads ${NTHREADS} -o eledream.264 
%s"
'''

def fluidanimate_splitdiff(tfile, n, frmax):
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
        newtarfilename = os.path.join(prefixfolder, tarfilename + '_'
                                      + '%02d' % (i+1) + '.tar')
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


def freqmine_splitequal(tfile, n, ms):
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
            newfilename = fm.name.split('.')[0] + '_' + ('%02d' % partscount) \
                          + '.' + fm.name.split('.')[1]
            fd = open(newfilename,'w')
        elif line%splitlen == 0 and line<splitlen*n:
            fd.close()
            newtarfilename = os.path.join(prefixfolder,tarfilename + '_'
                                          + ('%02d' % partscount) + '.tar')
            print(partscount,newtarfilename)
            tar2 = tarfile.open(newtarfilename,'w')
            tar2.add(newfilename)
            tar2.close()
            os.remove(newfilename)
            lfile.append((newtarfilename,newfilename))
            partscount += 1
            newfilename = fm.name.split('.')[0] + '_' + str(partscount) \
                          + '.' + fm.name.split('.')[1]
            fd = open(newfilename,'w')
        fd.write(linetxt.decode())
    newtarfilename = os.path.join(prefixfolder,tarfilename + '_'
                                  + ('%02d' % partscount) + '.tar')
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

def freqmine_splitdiff(tfile, n, ms):
    """
    Split Freqmine Benchmark input tar file within 'n' arithmetic progressive
    size parts of new tar files with names like originalname_1 ...
    originalname_n.

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
                newfilename = fm.name.split('.')[0] + '_' \
                              + ('%02d' % partscount) + '.' \
                              + fm.name.split('.')[1]
                fd = open(newfilename,'w')
            elif line%splitlen == 0 and line<=splitlenbase*(n-1):
                fd.close()
                newtarfilename = os.path.join(prefixfolder,tarfilename + '_'
                                              + ('%02d' % partscount) + '.tar')
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
    newtarfilename = os.path.join(prefixfolder,tarfilename + '_'
                                  + ('%02d' % partscount) + '.tar')
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

def dedup_splitdiff(tfile, n):
    """
    Split Dedup Benchmark input tar file within 'n' arithmetic progressive
    size parts of new tar files with names like originalname_1 ...
    originalname_n.

    :param tfile: Dedup input tar file name to split.
    :param n: Count of aritmetic progressive size split parts.
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
    f = tar.extract(fm)

    from bitstring import ConstBitStream
    bs = ConstBitStream(filename=fm.name)

    print('Splitting input file %s on %s files' % (tfile,n))
    partscount = 1

    nbytestotal = os.stat(fm.name).st_size
    splitlenbase = nbytestotal // n
    splitlen = splitlenbase

    fd = None
    lfile = []
    for parts in range(n):
        partscount = parts + 1
        splitlen = partscount * splitlenbase
        newfilename = fm.name.split('.')[0] \
                      + '_' + ('%02d' % partscount) + '.' \
                      + fm.name.split('.')[1]
        if partscount < n:
            fd = open(newfilename, 'wb')
            print(partscount, " splitlen: ", splitlen, "filename: ",
                  newfilename)
            bs.bytepos = splitlen
            bs2 = bs[:bs.pos]
            bs2.tofile(fd)
            fd.close()
        else:
            import shutil
            shutil.copy(fm.name, newfilename)
            print(partscount, " splitlen: ", splitlen, "filename: ",
                  newfilename)
            os.remove(fm.name)
        newtarfilename = os.path.join(prefixfolder, tarfilename + '_'
                                      + ('%02d' % partscount) + '.tar')
        print("  Tarfile: ", newtarfilename)
        tar2 = tarfile.open(newtarfilename, 'w')
        tar2.add(newfilename)
        tar2.close()
        os.remove(newfilename)
        lfile.append((newtarfilename, newfilename))
        bs.pos = 0

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
        fconf2.write(template_dedup_inputconf % (fnbase+'.runconf',fn))
        fconf2.close()
    print('Sucessful finished split operation. Total parts = ',partscount)


def x264_splitdiff(tfile, n):
    """
    Split x264 Benchmark input tar file within 'n' arithmetic progressive
    size parts of new tar files with names like originalname_1 ...
    originalname_n.

    :param tfile: Dedup input tar file name to split.
    :param n: Count of aritmetic progressive size split parts.
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
    f = tar.extract(fm)

    from bitstring import ConstBitStream
    bs = ConstBitStream(filename=fm.name)

    print('Splitting input file %s on %s files' % (tfile,n))
    partscount = 1

    nframes = 0
    nframestotal = 512
    splitlenbase = nframestotal // n
    splitlen = splitlenbase

    fd = None
    lfile = []
    eof = False
    while not eof:
        found = bs.find(b'FRAME', start=bs.pos + 1, bytealigned=True)
        if found:
            nframes += 1
            if nframes == 1:
                if partscount < n:
                    newfilenameparts = fm.name.split('_')
                    newfilename = newfilenameparts[0] + '_' + newfilenameparts[1] \
                              + '_' + ('%02d' % splitlen) + '.' \
                              + fm.name.split('.')[1]
                    fd = open(newfilename, 'wb')
                    print(partscount, " splitlen: ", splitlen, "filename: ",
                      newfilename)
                else:
                    print(partscount, " splitlen: ", splitlen, "filename: ",
                      fm.name)
                    newtarfilename = os.path.join(prefixfolder,
                                            tarfilename + '_'
                                            + ('%02d' % partscount) + '.tar')
                    print("  Tarfile: ", newtarfilename)
                    tar2 = tarfile.open(newtarfilename, 'w')
                    tar2.add(fm.name)
                    tar2.close()
                    os.remove(fm.name)
                    tar.close()
                    lfile.append((newtarfilename, fm.name))
                    eof = True
            elif nframes > splitlen:
                bs2 = bs[:bs.pos]
                bs2.tofile(fd)
                fd.close()
                newtarfilename = os.path.join(prefixfolder, tarfilename + '_'
                                              + ('%02d' % partscount) + '.tar')
                print("  Tarfile: ", newtarfilename)
                tar2 = tarfile.open(newtarfilename, 'w')
                tar2.add(newfilename)
                tar2.close()
                os.remove(newfilename)
                lfile.append((newtarfilename, newfilename))
                partscount += 1
                if partscount < n:
                    splitlen = partscount * splitlenbase
                else:
                    splitlen = nframestotal
                nframes = 0
                bs.pos = 0
        else:
            print("Erro: Frame %s não encontrado" % nframes)
            break

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
        fconf2.write(template_x264_inputconf % (fnbase+'.runconf',fn))
        fconf2.close()
    print('Sucessful finished split operation. Total parts = ',partscount)


def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    packagechoice = ['freqmine', 'fluidanimate', 'dedup', 'x264']
    splitchoice = ['equal', 'diff']
    parser = argparse.ArgumentParser(description='Script to split a parsec '
                                                 'input file on specific parts')
    parser.add_argument('-p','--package', help='Package name to be '
                                               'used on split.',
                        choices=packagechoice, required=True)
    parser.add_argument('-n','--numberofparts', help='Number of split parts',
                        type=int, required=True)
    parser.add_argument('-t','--typeofsplit', help='Split on equal or diferent '
                                                   'size partes parts',
                        choices=splitchoice, default='diff')
    parser.add_argument('-x', '--extraarg', help='Specific argument: Freqmine='
                                                 'minimum support (11000), '
                                                 'Fluidanimate=Max number of '
                                                 'frames',
                        type=int)
    parser.add_argument('inputfilename', help='Input filename from Parsec '
                                              'specificated package.')
    args = parser.parse_args()
    return args


def main():
    """
    Main function executed from console run.

    """

    args = argsparsevalidation()
    if os.path.isfile(args.inputfilename):
        if args.package == 'freqmine':
            if args.typeofsplit == 'diff':
                freqmine_splitdiff(args.inputfilename,
                                   args.numberofparts,args.extraarg)
            else:
                freqmine_splitequal(args.inputfilename,
                                    args.numberofparts,args.extraarg)
        elif args.package == 'fluidanimate':
            if args.typeofsplit == 'diff':
                fluidanimate_splitdiff(args.inputfilename,
                                       args.numberofparts,args.extraarg)
            else:
                fluidanimate_splitdiff(args.inputfilename,
                                       args.numberofparts,args.extraarg)
        elif args.package == 'dedup':
            if args.typeofsplit == 'diff':
                dedup_splitdiff(args.inputfilename,
                               args.numberofparts)
            else:
                print("Error: 'diff' type of part split only on dedup app.")
        elif args.package == 'x264':
            if args.typeofsplit == 'diff':
                x264_splitdiff(args.inputfilename,
                                       args.numberofparts)
            else:
                print("Error: 'diff' type of part split only on x264 app.")
    else:
        print("Error: File name not found.")
        exit(1)

if __name__ == '__main__':
    main()
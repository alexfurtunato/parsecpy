#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
    Module to process parsec logs files within a folder and generate
    a Pandas Dataframe with mesures times and speedups os runs.

    parsecpy_processlogs [-h] foldername outputfilename

    Script to parse a folder with parsec log files and save measures an output
    file

    positional arguments
        foldername
            Foldername with parsec log files.
        outputfilename
            Filename to save the measures dictionary.

    optional arguments
        -h, --help
            show this help message and exit
    Example
        parsecpy_processlogs logs_folder my-logs-folder-data.dat
"""

import os
import argparse

from parsecpy.dataprocess import ParsecLogsData

def argsparsevalidation():
    """
    Validation of script arguments passed via console.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to parse a folder '
                                                 'with parsec log files and '
                                                 'save measures to output file')
    parser.add_argument('foldername', help='Foldername with parsec log files.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from console run.

    """

    runfiles = []
    args = argsparsevalidation()
    if os.path.isdir(args.foldername):
        logs = ParsecLogsData(args.foldername)
        print("\nProcessing folder: ", logs.foldername)
        print(len(logs.runfiles),"files")
    else:
        print("Error: Folder name not found.")
        exit(1)

    if(logs.runfiles):
        print("\nProcessed Data: \n")
        print(logs)
        logs.savedata()
    else:
        print("Warning: Folder is empty")
        exit(1)

if __name__ == '__main__':
    main()
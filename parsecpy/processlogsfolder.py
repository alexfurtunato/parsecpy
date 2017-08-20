#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
  Module to process parsec output informations.

"""

import os
import argparse

from dataprocess import ParsecLogsData

def argsparsevalidation():
    """
    Validation of passed script arguments.

    :return: argparse object with validated arguments.
    """

    parser = argparse.ArgumentParser(description='Script to parse a folder with parsec log files and save measures to output file')
    parser.add_argument('foldername', help='Foldername with parsec log files.')
    args = parser.parse_args()
    return args

def main():
    """
    Main function executed from the linux shell script run.
    Process a folder with parsec log files and return Dataframes.

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

__version__ = '0.1'
__author__ = 'Igofed'

import argparse
import ImProc

import os

def argParse():

    folder = "data"
    parser = argparse.ArgumentParser(description="Ball detection")
    #output = ImProc.mkDir(dirName="temp")
    parser.add_argument("-folder", "--folder",
                        required=False,
                        type=str,
                        default= folder,
                        help = 'Folder with a set of images')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argParse()

    __mt = ImProc.ImProcess(folder = args.folder)


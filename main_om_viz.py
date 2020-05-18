#!/usr/bin/env python
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-01-01
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import argparse
import sys
import tool as tl
import om_viz as om_viz
#------------------------#
#     main function   
#     e.g., python main_fit.py [-h] [-s SEQFN] [-o OUTDIR] [-d on/off]
#------------------------#
if __name__ == "__main__":
    #--- arguments ---#
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",  "--outdir",    type=str, help="output dir")
    parser.add_argument("-a",  "--animation", type=int, help="animation (1:yes, 0:no)")
    args = parser.parse_args()
    if(len(sys.argv)<2):
        parser.print_help(); tl.error("parser")
    #--- check output dir ---#
    if(args.outdir!=None): 
        outdir = args.outdir
    else: parser.print_help(); tl.error("parser")
    #--- check animation ---#
    if(args.animation!=None): 
        animation = args.animation
    else: animation=tl.NO
    #--- print args ---#
    tl.comment('main_om_viz.py - args')
    print("-------------------------")
    print("outdir:[%s]"%outdir)
    print("plot animation (1:yes, 0:no):[%d]"%animation)
    print("-------------------------")

    #------------------------------#
    # Start VIZ
    #------------------------------#
    om_viz.run_viz(outdir, animation)
    sys.exit(0);




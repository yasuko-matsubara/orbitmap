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
import orbitmap as orbitmap
DBG=tl.YES
#------------------------#
#     main function   
#     e.g., python main_normalize.py [-h] [-i inputSEQFN] [-o OUTFN] [-l WINDOW (optional)]
#------------------------#
if __name__ == "__main__":
    #--- arguments ---#
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--seqfn", type=str, help="input seq filename")
    parser.add_argument("-w",  "--wd", type=int, help="sampling window size (optional)")
    parser.add_argument("-s",  "--swd", type=int, help="smooth window size (optional)")
    parser.add_argument("-o",  "--outfn", type=str, help="output seq filename")
    args = parser.parse_args()
    #--- check sequencefn ---#
    if(args.seqfn!=None):
        seqfn = args.seqfn
    else: parser.print_help(); tl.error("parser")
    #--- check window ---#
    if(args.wd!=None):
        wd = args.wd
    else: wd=1;
    if(args.swd!=None):
        swd = args.swd
    else: swd=1;
    #--- check output ---#
    if(args.outfn!=None): 
        outfn = args.outfn
    else: parser.print_help(); tl.error("parser")
    #--- print args ---#
    tl.comment('main_norm.py - args')
    lstep=swd; pstep=int(np.ceil(lstep*orbitmap.LP_R))
    print("-------------------------")
    print("seqfn:  [%s]"%seqfn)
    print("sampling window size: [%d]"%wd)
    print("smooth window size: [%d]"%swd)
    print("lstep: [%d], pstep: [%d]"%(lstep, pstep))
    print("outfn:  [%s]"%(outfn))
    print("-------------------------")

    #------------------------------#
    # LOAD DATA
    #------------------------------#
    data=tl.loadsq(seqfn).T
    (n,d) = np.shape(data)
    print("(n,d)=(%d,%d)"%(n,d))
    # avoid zero/nan (kf) 
    data+=tl.ZERO*np.random.rand(n,d)
    data=tl.smootherAVG(data,wd)
    data=tl.smoothMA(data, pstep)
    data=data[int(pstep):, :] 
    data=tl.normalizeZ(data)
    Xorg=data; # original data
    tl.save_txt(Xorg, '%s'%(outfn))
    sys.exit(0);




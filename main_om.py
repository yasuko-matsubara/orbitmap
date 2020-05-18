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
DBG=tl.NO
#------------------------#
#     main function   
#     e.g., python main_fit.py [-h] [-s SEQFN] [-o OUTDIR] [-d on/off]
#------------------------#
if __name__ == "__main__":
    #--- arguments ---#
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--iseqfn", type=str, help="input seqs filename")
    parser.add_argument("-l",  "--lstep", type=int, help="n-step-ahead")
    parser.add_argument("-o",  "--outdir",type=str, help="output dir")
    parser.add_argument("-t",  "--tst",   type=int, help="starting position (optional)")
    parser.add_argument("-n",  "--n",     type=int, help="duration (optional)")
    parser.add_argument("-m",  "--mdir",  type=str, help="model DB dir (optional)")
    parser.add_argument("-p",  "--mscale",type=int, help="multi-scale/process (optional)")
    args = parser.parse_args()
    if(len(sys.argv)<2):
        parser.print_help(); tl.error("parser")
    #--- check sequencefn ---#
    if(args.iseqfn!=None):
        iseqfn = args.iseqfn
    else: parser.print_help(); tl.error("parser")
    #--- check l-step-ahead ---#
    if(args.lstep!=None):
        lstep = args.lstep
    else: parser.print_help(); tl.error("parser")
    #--- check output dir ---#
    if(args.outdir!=None): 
        outdir = args.outdir
    else: parser.print_help(); tl.error("parser")
    #--- check tst (start time) ---#
    if(args.tst!=None): my_tst=args.tst
    else: my_tst=0
    #--- check n (duration) ---#
    if(args.n!=None): my_n=args.n
    else: my_n=tl.INF
    #--- check modelfn ---#
    if(args.mdir!=None):
        mdir= args.mdir
    else: mdir = ''
    #--- check multi-scale/process ---#
    if(args.mscale!=None):
        mscale= args.mscale
    else: mscale = 1 # single-scale
    #--- print args ---#
    tl.comment('main_om.py - args')
    tl.msg("-------------------------")
    tl.msg("input-seqfn: [%s]"%iseqfn)
    tl.msg("ls-step-ahead: [%d]"%lstep)
    tl.msg("multi-scale: [%d]"%mscale)
    tl.msg("outdir: [%s]"%outdir)
    tl.msg("model dir (scan): [%s]"%mdir)
    tl.msg("t_st (opt): [%s]"%my_tst)
    tl.msg("duration (opt): [%s]"%my_n)
    tl.msg("-------------------------")
    try:
        tl.mkdir("%s"%outdir)
    except:
        tl.error("cannot find: %s"%outdir)
    
    #------------------------------#
    # LOAD DATA
    #------------------------------#
    data=tl.loadsq(iseqfn).T
    data=data[my_tst:,:]
    (n,d) = np.shape(data)
    #--- set data length (my_n) optional ---#
    if(my_n < n): data=data[0:my_n,:]; (n,d) = np.shape(data)
    tl.msg("(n,d)=(%d,%d)"%(n,d))
    Xorg=data # original data
    tl.save_txt(Xorg, '%sXorg.txt'%(outdir))
    #------------------------------#
    # OrbitMap 
    #------------------------------#
    #--- est (pre-processing) or scan (streaming) ---#
    if(mdir==''):
        # start estimation
        orbitmap.run_est(Xorg, lstep, outdir, mscale)
    else:
        # start scan
        orbitmap.run_scan(Xorg, lstep, mdir, outdir, mscale)
    #------------------------------#
    sys.exit(0);



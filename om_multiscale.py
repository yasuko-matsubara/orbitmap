#!/usr/bin/env python
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-03-16
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
import numpy as np
import tool as tl
import om_mdb as om_mdb
import orbitmap as orbitmap
MULTI=tl.YES # multiprocessing (default: YES)
DBG=tl.NO 

#--------------------------------#
#    run est (pre-processing) / multi-scale 
#--------------------------------#
def run_est_mscale(Xorg, lstep, mscale_h, outdir):
    tl.msg("multiscale (run_est_mscale) start... (multiprocessing: %d)"%MULTI)
    # (1) create multi-scale sequences
    (XHs, wdHs, outdirHs) = _create_multiscale_seqs(Xorg, lstep, mscale_h, outdir)
    # (2) arg-setting 
    arg_list=[]
    for i in range(0,mscale_h): arg_list.append([XHs[i], lstep, outdirHs[i], wdHs[i]])
    # (3) start orbitmap (pre-processing) 
    if(MULTI): 
        _multi_proc_run_est(arg_list) 
    else: 
        for i in range(0,mscale_h): orbitmap.run_est_single(XHs[i], lstep, outdirHs[i], wdHs[i])
    tl.msg('multiscale (run_est_mscale) end.')
#--------------------------------#
#--------------------------------#
def _multi_proc_run_est(arg_list):
    n_proc=len(arg_list)
    pool = tl.multiprocessing.Pool(processes=n_proc)
    return pool.map(_wrapper_run_est, arg_list)
def _wrapper_run_est(arg_list):
    return orbitmap.run_est_single(*arg_list)
#-------------------------------------------------#

#--------------------------------#
#    run scan (streaming)/ multi-scale
#--------------------------------#
def run_scan_mscale(Xorg, lstep, mscale_h, modeldir, outdir):
    tl.msg("multiscale (run_scan_mscale) start... (multiprocessing: %d)"%MULTI)
    # (1) create multi-scale sequences
    (XHs, wdHs, outdirHs) = _create_multiscale_seqs(Xorg, lstep, mscale_h, outdir)
    # (2) arg-setting
    mdirHs={}; arg_list=[]
    for i in range(0,mscale_h): 
        mdirHs[i]='%s_Level%d/'%(modeldir,i) 
        arg_list.append([XHs[i], lstep, mdirHs[i], outdirHs[i], wdHs[i]])
    # (3) start orbitmap (streaming)
    if(MULTI): 
        _multi_proc_run_scan(arg_list) 
    else: 
        for i in range(0,mscale_h): orbitmap.run_scan_single(XHs[i], lstep, mdirHs[i], outdirHs[i], wdHs[i])
    tl.msg("multiscale (run_scan_mscale) end.")
    #--------------------------------#
def _multi_proc_run_scan(arg_list):
    n_proc=len(arg_list)
    pool = tl.multiprocessing.Pool(processes=n_proc)
    return pool.map(_wrapper_run_scan, arg_list)
def _wrapper_run_scan(arg_list):
    return orbitmap.run_scan_single(*arg_list)
#-------------------------------------------------#


#--------------------------------#
#    create multi-scale sequences (mscale_h-levels)
#--------------------------------#
def _create_multiscale_seqs(Xorg, lstep, mscale_h, outdir):
    tl.msg("_create_multiscale_seq (lstep: %d, mscale_h: %d)"%(lstep, mscale_h))
    tl.save_seq_txt_pdf(Xorg, "%sXorg"%outdir)
    # multi-scale sequences
    XHs={}; Xagg=tl.dcopy(Xorg); outdirHs={}
    wdHs={} # window size list
    for i in range(0,mscale_h):
        # window size at i-th level
        if(i<mscale_h-1): wdHs[i]=lstep*(2**(mscale_h-1-i)) # if HT=4: 3,2,1 ... (i.e., 8xlstep, 4xlstep, 2xlstep, 1)
        #if(i<mscale_h-1): wdHs[i]=lstep*(2**(mscale_h-2-i))  # if HT=4: 2,1,0 ... (i.e., 4xlstep, 2xlstep, lstep, 1)
        #if(i<mscale_h-1): wdHs[i]=lstep*(2**(mscale_h-3-i)) # if HT=4: 1,0,-1 ... (i.e., 2xlstep, lstep, 1/2xlstep, 1
        else: wdHs[i]=1 # if, last seq
    for i in range(0,mscale_h):
        XHs[i]=tl.smoothWMAo(Xagg, wdHs[i]) # compute i-th level
        Xagg=Xagg-XHs[i] # delete current scale
    for i in range(0,mscale_h):
        #XHs[i]= XHs[i][int(wdHs[0]):,:] # delete longest-window
        XHs[i][0:int(wdHs[0]),:]=tl.NAN # delete longest-window
        outdirHs[i]='%s_Level%d/'%(outdir,i); tl.mkdir(outdirHs[i]) # directory
        # save sequence at i-th level
        tl.save_txt(XHs[i], '%sXorg.txt'%(outdirHs[i])) # save data
        tl.plt.clf(); tl.plt.plot(XHs[i]) # plot and save figure
        tl.savefig("%sXorg"%(outdirHs[i]), 'pdf')
    tl.msg("wdHs: ")
    tl.msg(wdHs)
    return (XHs, wdHs, outdirHs)



#---------------#
#     main      #
#---------------#
if __name__ == "__main__":

    tl.msg("om_multiscale")
    

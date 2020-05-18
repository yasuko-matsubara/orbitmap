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
import numpy as np
import tool as tl
import time
import om_trans as om_trans
try:
    import lmfit
except:
    tl.error("can not find lmfit - please see http://lmfit.github.io/lmfit-py/")

#-----------------------------#
# lmfit (default)
#XTL=1.e-8
#FTL=1.e-8
#MAXFEV=100 
# lmfit (efficient)
XTL=0.1
FTL=0.1
MAXFEV=20
#-----------------------------#
DBG=tl.NO #YES
#-----------------------------#

#----------------------------#
#    model fitting           #
#----------------------------#
# ftype: forward/backward
# dictMS={'cid': cid, 'si': si, 'rho': rho, 'smin': smin, 'errV': errV}
def fit(Xe, mdb, ftype, dictMS, wtype='uniform', dps=1):
    #--------------------------------#
    if(DBG): tl.comment("=== start LMfit ===")
    if(DBG): tl.comment("(%s,%s,%d)"%(ftype, wtype, dps)) 
    if(DBG): tl.msg("rmse: %f"%_distfunc_rmse(Xe, mdb, dictMS))
    if(DBG): tl.comment("start fitting") 
    #--------------------------------#
    if(DBG): tic = time.clock()
    (dictMS)=_fit(Xe, mdb, ftype, dictMS, wtype, dps)
    if(DBG): toc = time.clock(); fittime= toc-tic;
    dictMS['errV']= _distfunc_rmse(Xe, mdb, dictMS)
    #--------------------------------#
    if(DBG): tl.comment("end fitting")
    if(DBG): tl.msg("time: %f"%fittime)
    if(DBG): tl.comment("=== end LMfit ===")
    #--------------------------------#
    return dictMS 


def _fit(Xe, mdb, ftype, dictMS, wtype, dps):
    #---------------------------------------# 
    # (1) create param set
    P=_createP(dictMS, ftype)
    #---------------------------------------# 
    # (2) start lmfit
    lmsol = lmfit.Minimizer(_distfunc, P, fcn_args=(Xe, mdb, dictMS, ftype, wtype, dps)) 
    res=lmsol.leastsq(xtol=XTL, ftol=FTL, maxfev=MAXFEV)
    if(DBG): tl.msg("end")
    #---------------------------------------# 
    # (3) update param set
    dictMS=_updateP(res.params, dictMS, ftype)
    #---------------------------------------# 
    # (4) generate
    return (dictMS) 

#----------------------------#
def _createP(dictMS, ftype):
    #--------------------------------------------------#
    si=dictMS['si']; rho=dictMS['rho']
    #--------------------------------------------------#
    P = lmfit.Parameters()
    k=len(si)
    V=True
   #--------------------------------------------------#
    if(ftype=='forward'):
    #--------------------------------------------------#
        for i in range(0,k):
            P.add('si_%i'%(i), value=si[i], vary=V)
        #P.add('rho', value=rho, vary=V)
    #--------------------------------------------------#
    return P
#----------------------------#
def _updateP(P, dictMS, ftype):
    #--------------------------------------------------#
    k=len(dictMS['si'])
    #--------------------------------------------------#
    if(ftype=='forward'):
    #--------------------------------------------------#
        for i in range(0,k):
            dictMS['si'][i]=P['si_%i'%(i)].value
        #dictMS['rho']=P['rho'].value
    return dictMS
#----------------------------#

#--------------------------------------#
#    objective functions  #
#    return the array to be minimized
#--------------------------------------#
def _distfunc(P, Xe, mdb, dictMS, ftype, wtype, dps): #data, nlds, ftype, wtype, dps):
    if(DBG): tl.dotting()
    if(DBG): tl.msg("_distfunc: rmse:%f, rho:%f"%(_distfunc_rmse(Xe, mdb, dictMS), dictMS['rho']))
    (n, d)=np.shape(Xe); lene=n; 
    # update parameter set
    dictMS=_updateP(P, dictMS, ftype)
    # generate seq
    (Oe, Se, Re) = om_trans.gen_forward(mdb, dictMS, lene, dps)
    if(dps>1): Xe=Xe[range(0,lene,dps),:]
    # diffs
    diff=Xe.flatten() - Oe.flatten()
    diff[np.isnan(diff)]=0
    # weighted-fit
    diff=diff*tl.func_Weight(len(diff), wtype)
    return diff
def _distfunc_rmse(Xe, mdb, dictMS):
    (n,d) = np.shape(Xe); lene=n;
    (Oe, Se, Re) = om_trans.gen_forward(mdb, dictMS, lene)
    return tl.RMSE(Xe, Oe);

 

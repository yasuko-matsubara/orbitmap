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
try:
    import lmfit
except:
    tl.error("can not find lmfit - please see http://lmfit.github.io/lmfit-py/")

#-----------------------------#
# lmfit (default)
XTL=1.e-8
FTL=1.e-8
MAXFEV=100 
# lmfit (incremental)
XTLi=0.1
FTLi=0.1
MAXFEVi=20
#-----------------------------#
DBG=tl.NO
#-----------------------------#

#----------------------------#
#    model fitting           #
#----------------------------#
# ftype: [si / A2 / A01 / B01]
def nl_fit(nlds, ftype, wtype, dps): 
    nlds_org=tl.dcopy(nlds)
    #--------------------------------#
    if(ftype!='si' and ftype!='A2'and ftype!='A01' and ftype!='B01'): 
        tl.warning("fit.py: usage: [si/M2/A01/B01]")
    #--------------------------------#
    if(ftype=='si'): # if si, i.e., arg{s(0)=si}
        global MAXFEV; MAXFEV=MAXFEVi
        global XTL; XTL=XTLi
        global FTL; FTL=FTLi
    #--------------------------------#
    if(DBG): tl.comment("=== start LMfit ===")
    if(DBG): tl.comment("(%s,%s,%d)"%(ftype, wtype, dps)) 
    if(DBG): tl.msg("rmse: %f"%_distfunc_rmse(nlds))
    if(DBG): tl.comment("start fitting") 
    #--------------------------------#
    if(DBG): tic = time.clock()
    nlds=_nl_fit(nlds, ftype, wtype, dps) 
    if(DBG): toc = time.clock(); fittime= toc-tic;
    #--------------------------------#
    if(_distfunc_rmse(nlds_org)<_distfunc_rmse(nlds)): nlds=nlds_org #notfin
    if(DBG): tl.comment("end fitting")
    if(DBG): tl.msg("time: %f"%fittime)
    if(DBG): tl.msg("rmse: %f"%_distfunc_rmse(nlds))
    if(DBG): tl.comment("=== end LMfit ===")
    #--------------------------------#
    return nlds 


def _nl_fit(nlds, ftype, wtype, dps): 
    #---------------------------------------# 
    # (1) create param set
    P=_createP(nlds, ftype)
    #---------------------------------------# 
    # (2) start lmfit
    lmsol = lmfit.Minimizer(_distfunc, P, fcn_args=(nlds.data, nlds, ftype, wtype, dps))
    res=lmsol.leastsq(xtol=XTL, ftol=FTL, maxfev=MAXFEV)
    if(DBG): tl.msg("end")
    #---------------------------------------# 
    # (3) update param set
    nlds=_updateP(res.params, nlds, ftype)
    #---------------------------------------# 
    return nlds 


#----------------------------#
def _createP(nlds, ftype):
    P = lmfit.Parameters()
    #PARAM_MX=0.1
    #pm=PARAM_MX
    #PARAM_INI=1.e-4 #6
    k=nlds.k; d=nlds.d
    V=True
   #--------------------------------------------------#
    if(ftype=='si'):
    #--------------------------------------------------#
        for i in range(0,k):
            P.add('si_%i'%(i), value=nlds.si[i], vary=V)
   #--------------------------------------------------#
    if(ftype=='A2'):
    #--------------------------------------------------#
        for i in range(0,k):
            P.add('A2_%i'%(i), value=nlds.A2[i][i][i]) 
            #P.add('A2_%i'%(i), value=PARAM_INI, min=-pm,max=+pm,vary=V)
    #--------------------------------------------------#
    if(ftype=='A01'):
    #--------------------------------------------------#
        for i in range(0,k):
            P.add('A0_%i'%(i), value=nlds.A0[i], vary=V) 
            for j in range(0,k):
                P.add('A1_%i_%i'%(i,j), value=nlds.A1[i][j], vary=V) #notfin
    #--------------------------------------------------#
    if(ftype=='B01'):
    #--------------------------------------------------#
        for i in range(0,d):
            P.add('B0_%i'%(i), value=nlds.B0[i])
            for j in range(0,k):
                P.add('B1_%i_%i'%(i,j), value=nlds.B1[i][j]) 
    #--------------------------------------------------#
    return P
#----------------------------#
def _updateP(P, nlds, ftype):
    k=nlds.k; d=nlds.d
    #--------------------------------------------------#
    if(ftype=='si'):
    #--------------------------------------------------#
        for i in range(0,k):
            nlds.si[i]=P['si_%i'%(i)].value
    #--------------------------------------------------#
    if(ftype=='A2'):
    #--------------------------------------------------#
        for i in range(0,k):
            nlds.A2[i][i][i]=P['A2_%i'%(i)].value
    #--------------------------------------------------#
    if(ftype=='A01'):
    #--------------------------------------------------#
        for i in range(0,k):
            nlds.A0[i]=P['A0_%i'%(i)].value
            for j in range(0,k):
                nlds.A1[i][j]=P['A1_%i_%i'%(i,j)].value
    #--------------------------------------------------#
    if(ftype=='B01'):
    #--------------------------------------------------#
        for i in range(0,d):
            nlds.B0[i]=P['B0_%i'%(i)].value
            for j in range(0,k):
                nlds.B1[i][j]=P['B1_%i_%i'%(i,j)].value
    #--------------------------------------------------#
    return nlds
#----------------------------#

#--------------------------------------#
#    objective functions  #
#    return the array to be minimized
#--------------------------------------#
def _distfunc(P, data, nlds, ftype, wtype, dps):
    if(DBG): tl.dotting()
    n=np.size(data,0)
    # update parameter set
    nlds=_updateP(P,nlds, ftype)
    # generate seq
    (Sta, Obs)=nlds.gen(n, dps) 
    if(dps>1): data=data[range(0,n,dps),:]
    # diffs
    diff=data.flatten() - Obs.flatten()
    diff[np.isnan(diff)]=0
    # weighted-fit
    diff=diff*tl.func_Weight(len(diff), wtype)
    return diff
def _distfunc_rmse(nlds):
    data=nlds.data
    (Sta, Obs)=nlds.gen(len(data)) 
    return tl.RMSE(data, Obs);



 

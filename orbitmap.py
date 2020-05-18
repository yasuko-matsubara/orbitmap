#!/usr/bin/env python
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-04-24
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
import numpy as np
import tool as tl
import nlds as nl
import om_trans as om_trans
import om_mdb as om_mdb
import om_viz as om_viz
import om_multiscale as om_multiscale
#------------------------------------------------#
# --- debug
COMMENT=tl.NO
DBG0=tl.NO 
DBG1=tl.NO 
# --- I/O
TSAVE=tl.NO #YES # save every time tick (default: NO)
SAVE_TRIALS=tl.NO #YES # plot trials
#------------------------------------------------#
# --- windows 
LMIN=om_mdb.LMIN # minimum window length 
LMAX=om_mdb.LMAX # maximum window length 
LM_R=(1.0/1.0) # lm=ls*(1/1)--- minimum window length rate
LP_R=(1.0/5.0) # lp=ls*(1/5) --- sliding/reporting-windowrate
SMIN_R=LM_R #1.0 # minimum length of (multi-transitions)

# --- regimeID settings
REID_NULL=om_mdb.REID_NULL #regimeID (null: [-1])
REID_CPT=om_mdb.REID_CPT   #regimeID (cpt: [-1])
# --- LMfit 
W_RR=om_mdb.W_RR # weighted LMfit for RR 
W_RE=om_mdb.W_RE # weighted LMfit for RE
W_FM=om_mdb.W_FM # weighted LMfit for FM 
# seed, init rho
RHO_INIT=0.1  # if, want to estimate, RHO_INIT=-1
AVGR=1.0 
AVG_TRIAL=10 
# bottom-up
EARLYSTOP=tl.NO  # default: NO
RHO_ITER_R_BU=1.2 #:1.5 
MINTRIAL=5; MAXTRIAL=10 #6 #10
# check unfit patterns
WANT_AVOID_UNFIT=tl.YES
AVOIDUNFIT_R=1.5 
# cut&paste
CUTNP="LAST" #LAST/MEAN (cut&paste: using last or mean value)
RHO_MIN_R=2 # minimum rho ratio
# print/output
IDT_E="\t"
IDT_F="\t\t"
# refinement
AP_MD=0; AP_MS=0 #1 
#
OPTCNT=LMIN  # _find_opt_cut, search for opt cut-points (OPTCNT times) 
# multi-trans settings
MULTI_TRANS=tl.NO
# forecast
USE_Vb=tl.YES    # hop-step-jump-forecast (use "Vb",Vp,Vc)
F_WANT_OPT=tl.NO # optimize multi-step-fit
FSTART_R=2 # start forecast at tc=ls*FSTART_R 
# DPS
DPS=1 # dynamic point set ([1:lstep], default: 1)
#-------------------------------------------------#
# initialize env
def _set_env(lstep,mscale):
    # set DPS
    if(lstep>10): 
        global DPS; DPS=int(lstep/10) 
#-------------------------------------------------#



###################################################
#-------------------------------------------------#
#-------------------------------------------------#
def run_est(Xorg, lstep, outdir, mscale_h=1):
    tl.msg("run_est - start ...")
    _set_env(lstep, mscale_h)
    if(mscale_h==1): # single
        run_est_single(Xorg, lstep, outdir)
    else: # multi-scale
        om_multiscale.run_est_mscale(Xorg, lstep, mscale_h, outdir)
    tl.msg('run_est - end.')
    #--------------------------------#
def run_scan(Xorg, lstep, modeldir, outdir, mscale_h=1):
    tl.msg("run_scan - start ...")
    _set_env(lstep, mscale_h)
    if(mscale_h==1): # single 
        run_scan_single(Xorg, lstep, modeldir, outdir)
    else: # multi-scale
        om_multiscale.run_scan_mscale(Xorg, lstep, mscale_h, modeldir, outdir)
    tl.msg('run_scan - end.')
    #--------------------------------#
#-------------------------------------------------#
#-------------------------------------------------#
###################################################


#--------------------------------#
#    run scan (streaming) 
#--------------------------------#
def run_scan_single(Xorg, lstep, modeldir, outdir, wd_level=1):
    tl.msg("(%d) run_scan_single: start ..."%wd_level)
    mdb=tl.load_obj('%sMDB.obj'%(modeldir))
    (mdb, Snaps) = _OrbitMap(Xorg, lstep, wd_level, mdb, outdir) 
    return (mdb, Snaps)
#--------------------------------#
#--------------------------------#
#    run est (pre-processing) 
#--------------------------------#
def run_est_single(Xorg, lstep, outdir, wd_level=1):
    #--------------------------------#
    want_refinement=tl.YES
    #--------------------------------#
    mdb_cb=om_mdb.MDB('%s'%outdir)  # create modelDB
    Snaps_cb=init_Snaps(Xorg, lstep, wd_level) # SnapShots 
    mdb_cb=mdb_cb.update_Xminmax(Xorg) 
    errF_cb=tl.INF #; Snaps_cb=[];
    if(RHO_INIT==-1): rho_iter=_compute_rho_min(Xorg, lstep)
    else: rho_iter=RHO_INIT
    ERRs_full=[]; ERRs_cb_i=[]    
    outdir_tr="%s_trials/"%outdir
    if(SAVE_TRIALS): tl.mkdir(outdir_tr) # create directory
    tl.msg("(%d) run_est_single: start trial (find opt rho) ..."%wd_level)
    for i_trial in range(0,MAXTRIAL):
        #--------------------------------#
        mdb_i=tl.dcopy(mdb_cb); Snaps_i=tl.dcopy(Snaps_cb)
        #--------------------------------#
        # set rho, cleansing (e.g., remove inappropriate regimes), etc. 
        mdb_i.update_rho(rho_iter)
        (mdb_i, Snaps_i) =  mdb_i.update_apoptosis(Snaps_i, AP_MD, AP_MS, want_refinement)
        mdb_i = mdb_i.init_objects()
        #--------------------------------#
        # start OrbitMap
        (mdb_i, Snaps_i) =_OrbitMap(Xorg, lstep, wd_level, mdb_i, "%sTrial_%d_"%(outdir_tr, i_trial), SAVE_TRIALS, SAVE_TRIALS) #tl.YES) #tl.NO)
        #--------------------------------#
        if(i_trial > 0 and Snaps_i['errF_half']<errF_cb):  # update best-fit (if trial > 0)
            errF_cb=Snaps_i['errF_half']; mdb_cb=tl.dcopy(mdb_i); Snaps_cb=tl.dcopy(Snaps_i)
        #--------------------------------#
        tl.msg("(mscale-wd: %d) Trial %d rho=%f, c=%d, errE: %f, errF(half):%f, errF(full):%f"%(wd_level, i_trial, rho_iter, mdb_i.get_c(), Snaps_i['errE'], Snaps_i['errF_half'], Snaps_i['errF']))
        ERRs_full.append([i_trial, rho_iter, mdb_i.get_c(),  Snaps_i['errE'],  Snaps_i['errF_half'],  Snaps_i['errF']])
        tl.save_txt(ERRs_full, "%s_trials_results.txt"%(outdir))
        #--------------------------------#
        if(EARLYSTOP and Snaps_i['errF_half']>=errF_cb and i_trial>=MINTRIAL): break # if, no more better-fit, break
        if(mdb_i.get_c()<=1 and i_trial > 0): break # if, # of regimes is less than one, then, stop
        #--------------------------------#
        #--------------------------------#
        rho_iter*=RHO_ITER_R_BU # rho: try larger value 
        #--------------------------------#
    # final-best-fit
    (mdb_cb, Snaps_cb) =  mdb_cb.update_apoptosis(Snaps_cb, AP_MD, AP_MS, want_refinement)
    mdb_cb = mdb_cb.init_objects()
    if(Snaps_cb['errF_half'] > Snaps_cb['errS']): mdb_cb.CC=tl.NO # if, not approp. (no causalchain), ignore model
    (mdb_cb, Snaps_cb) =_OrbitMap(Xorg, lstep, wd_level, mdb_cb,  "%s"%(outdir), tl.YES, tl.YES)
    tl.msg("best est: c=%d, errE: %f, errF(half):%f, errF(full):%f"%( mdb_cb.get_c(), Snaps_cb['errE'], Snaps_cb['errF_half'], Snaps_cb['errF']))
    return (mdb_cb, Snaps_cb)
#--------------------------------#



#--------------------------------#
# OrbitMap 
#--------------------------------#
# (input)
# Xorg (original data stream) 
# lstep (lstep-ahead-forecast)
# wd_level (window size (multi-scale))
# mdb   (model parameter set) 
#--------------------------------#
# (output)
# mdb   (model parameter set) 
# Snaps (snapshots, etc.,... )
#--------------------------------#
def _OrbitMap(Xorg, lstep, wd_level, mdb, outdir, save_fig=True, save_full=True):
    #--------------------------------#
    (n,d)=np.shape(Xorg)
    Snaps=init_Snaps(Xorg, lstep, wd_level) # SnapShots 
    CGs=[] # for CGraph_snap_shots
    n=Snaps['n']; d=Snaps['d']; pstep=Snaps['pstep']
    PE_vc=_initP() # current best
    PE_vp=_initP() # previous pattern
    PE_vb=_initP() # previous-previous pattern
    # init time-ticks
    tc=pstep; PE_vc['tm_st']=0; PE_vc['tm_ed']=tc;
    if(COMMENT): tl.msg("_OrbitMap X(%d,%d), lstep=%d, pstep=%d, wd_level=%d, c=%d, rho=%f, dps=%d"%(n,d, lstep, pstep, wd_level, mdb.get_c(), mdb.rho_RE, DPS))
    #--------------------------------#
    while(True):
        #===================#
        tic = tl.time.clock()
        #===================#
        if(tc>=n): break
        if( np.isnan( Xorg[PE_vc['tm_st']:PE_vc['tm_ed'],:] ).sum()>0 ): PE_vc['tm_st']=tc; tc+=pstep; continue # if, nan-value, then ignore
        if(DBG0): tl.comment("tc:%d"%(tc))
        mdb=mdb.update_Xminmax(Xorg[(tc-pstep):tc,:]) # update Xmin, Xmax
        #--------------------------------#
        # 1. O-estimator 
        (PE_vb, PE_vp, PE_vc, mdb, Snaps) = _O_estimator(tc, Xorg, PE_vb, PE_vp, PE_vc, mdb, Snaps)
        tic2 = tl.time.clock() 
        if(DBG1): tl.msg("time(estimate):%f"%(tl.time.clock()-tic)) 
        #--------------------------------#
        if(tc<=lstep*FSTART_R or tc+lstep>=n): tc+=pstep; continue 
        #--------------------------------#
        # 2. O-generator 
        (Snaps) = _O_generator(tc, Xorg, PE_vb, PE_vp, PE_vc, mdb, Snaps)
        #===================#
        if(DBG1): tl.msg("time(forecast):%f"%(tl.time.clock()-tic2)) 
        toc = tl.time.clock(); fittime= toc-tic; 
        if(DBG1): tl.msg("time:%f"%(fittime))
        #if(save_full and TSAVE and outdir != ''): tl.save_mat(Snaps, "%sSnaps"%(outdir)) 
        Snaps['T_full'][tc]=fittime
        CGs.append(mdb.create_CGraph_rcds())
        # continue 
        tc+=pstep
        #===================#
    #--------------------------------#
    # final (SAVE) 
    #--------------------------------#
    if(mdb.CC is tl.NO): Snaps['Ve_full']=Snaps['Xorg']
    Snaps=compute_Snaps_Errs(Snaps)
    want_refinement=tl.NO
    (mdb, Snaps) =  mdb.update_apoptosis(Snaps, 0, 0, want_refinement) # if there's any un-used regime, then delete (but, no-refinement)
    # save results 
    if(save_full and outdir != ''):
        tl.save_mat(Snaps, "%sSnaps"%(outdir)) 
        tl.save_obj(mdb, "%sMDB"%(outdir))
        tl.save_obj(CGs, "%sCGs"%(outdir))
    if(save_fig and outdir!=''): 
        om_viz.saveResults_txt(Snaps, mdb, outdir)
        om_viz.plotResultsE(Snaps, mdb, outdir)
        om_viz.plotResultsF(Snaps, mdb, outdir)
        om_viz.plotCG(mdb, outdir)
    #--------------------------------#
    # return mdb, Snaps
    #--------------------------------#
    return (mdb, Snaps)



 

def init_Snaps(Xorg, lstep, wd_level):
    pstep=int(np.ceil(lstep*LP_R)) # set window (pstep)
    (n,d)=np.shape(Xorg)
    Snaps={'lstep':lstep, 'pstep':pstep, 'wd_level':wd_level, 
            'n':n, 'd':d, 'Xorg': Xorg, 'DPS': DPS,  
            'Xe':[], 'Ve':[], 'Se':[], 'Re':[], 'Te':[], 
            #'Se_full':np.nan*np.zeros((n,d+nl.KMAXPLS)), 'Sf_full':np.nan*np.zeros((n,d+nl.KMAXPLS)), 
            'Ve_full':np.nan*np.zeros((n,d)), 'Vf_full':np.nan*np.zeros((n,d)), 
            'Re_full':np.nan*np.zeros((n)), 'Rf_full':np.nan*np.zeros((n)),
            'Es_full':[], 'Ex_full':[], 'Ee_full':[], 'Ef_full':[], 
            'errX':np.nan, 'errE':np.nan, 'errF':np.nan, 'errF_half':np.nan,  
            #'cpts':[],
            'T_full': np.nan*np.zeros((n))}
    return Snaps

def compute_Snaps_Errs(Snaps):
    Xorg=Snaps['Xorg']; lstep=Snaps['lstep']; pstep=Snaps['pstep'];
    n=Snaps['n']; d=Snaps['d']
    Xsft=np.append(np.zeros((lstep+pstep,d)), Xorg[:-lstep-pstep,:],axis=0) # shifted-Xorg (ls+lp)
    Snaps['Es_full']=tl.RMSE_each(Xsft, Xorg)
    Snaps['Ex_full']=tl.RMSE_each(0*Xorg, Xorg)
    Snaps['Ee_full']=tl.RMSE_each(Snaps['Ve_full'], Xorg)
    Snaps['Ef_full']=tl.RMSE_each(Snaps['Vf_full'], Xorg)
    Snaps['errS']=tl.mynanmean(Snaps['Es_full'][lstep:])
    Snaps['errX']=tl.mynanmean(Snaps['Ex_full'][lstep:])
    Snaps['errE']=tl.mynanmean(Snaps['Ee_full'][lstep:])
    Snaps['errF']=tl.mynanmean(Snaps['Ef_full'][lstep:])
    Snaps['errF_half']=tl.mynanmean(Snaps['Ef_full'][int(n*0.5):]) 
    return Snaps


#--------------------------------#
# _initP()
#--------------------------------#
# PE_vb, PE_vp, PE_vc
#--------------------------------#
# md (model param set)
# cid (modelID)
# Vc (estimated events)
# siset (init set)
# tm_st (starting position)
# tm_ed (ending position)
# errV  (error, ||Vc-Xc||)
#--------------------------------#
def _initP():
    P={'md':[], 'cid':-1, 'Vc':[], 'siset':{}, 'tm_st':-1, 'tm_ed':-1, 'errV':tl.INF}
    return P
def _isNullP(P):
    if(P['md']==[]): return True
    #if(P['cid']==-1): return True
    else: return False


#--------------------------------#
# _RR, _RE, _RU
#--------------------------------#
# _RR: regime-reader
# _RE: regime-estimate 
# _RU: regime-update
#--------------------------------#
# Xc (current window)
# mdb (full model set)
# PE_cr (current parameter set)
#--------------------------------#
# regime-reader
def _RR(Xc, mdb, PE_cr):
    tm_st=PE_cr['tm_st']; tm_ed=PE_cr['tm_ed']
    Si=PE_cr['siset']
    (Vc, md_c, Si, cid)=mdb.search_md(Xc, Si, DPS)
    if(md_c==[]): 
        if(DBG0): tl.msg("%s ...... _RR: mdb (null)"%(IDT_E))
    else:
        (Sc,Vc)=md_c.gen()   # generate events 
        errV=tl.RMSE(Xc,Vc) 
        md_c.fn="seg:t%d-%d_e%.2f"%(tm_st, tm_ed, errV)
        if(DBG0): tl.msg("%s ...... _RR: errV:%f (cid:%d)"%(IDT_E, errV, cid))
        PE_cr={'md':md_c, 'cid':cid, 'Vc':Vc, 'Sc':Sc, 'siset':Si, 'tm_st':tm_st, 'tm_ed':tm_ed, 'errV':errV}
    return PE_cr
# regime-estimate
def _RE(Xc, tm_st, tm_ed):
    # estimate new model dynamics (md_c) 
    md_c=nl.NLDS(Xc, "t%d-%d"%(tm_st, tm_ed))
    md_c=md_c.fit(W_RE)  # lmfit
    (Sc,Vc)=md_c.gen()   # generate events 
    errV=tl.RMSE(Xc,Vc) 
    md_c.fn="seg:t%d-%d_e%.2f"%(tm_st, tm_ed, errV)
    if(DBG0): tl.msg("%s ...... _RE: errV:%f (cid:-1)"%(IDT_E, errV))
    PE_cr={'md':md_c, 'cid':-1, 'Vc':Vc, 'Sc':Sc, 'siset':{}, 'tm_st':tm_st, 'tm_ed':tm_ed, 'errV':errV}
    return PE_cr 
# regime-update
def _RU(Xc, PE_cr, fixORopt):
    tm_st=PE_cr['tm_st']; tm_ed=PE_cr['tm_ed']
    if(_isNullP(PE_cr)):
        if(DBG0): tl.msg("%s ...... _RU: (null)"%(IDT_E))
    else:
        cid=PE_cr['cid']               # put cid
        md_c=tl.dcopy(PE_cr['md'])     # copy model
        md_c.n=len(Xc); md_c.data=Xc   # copy data
        if(fixORopt=='OPT'):
            md_c=md_c.fit_si(W_RR, DPS) # update init si (i.e., s(0)=si)
        (Sc,Vc)=md_c.gen()              # generate events 
        errV=tl.RMSE(Xc,Vc) 
        md_c.fn="seg:t%d-%d_e%.2f"%(tm_st, tm_ed, errV)
        if(DBG0): tl.msg("%s ...... _RU: errV:%f (cid:%d)"%(IDT_E, errV, cid))
        PE_cr={'md':md_c, 'cid':cid, 'Vc':Vc, 'Sc':Sc, 'siset':{}, 'tm_st':tm_st, 'tm_ed':tm_ed, 'errV':errV}
    return PE_cr

# regime-trans-update
# PE_cr => {PE_vc : PE_vs}
def _RT(Xc, mdb, smin, PE_cr):
    PE_vc=tl.dcopy(PE_cr); PE_vs=[]; lenc=len(Xc) 
    dictMS={'cid':PE_cr['cid'], 'si':tl.dcopy(PE_cr['md'].si), 'rho':mdb.rho_MS, 'smin':smin, 'errV':tl.INF} 
    (Ve, Se, Re)=om_trans.gen_forward(mdb, dictMS, lenc)
    errV=tl.RMSE(Xc, Ve[0:lenc])
    #tl.eprint("_RT: before:%f -> after:%f"%(PE_cr['errV'], errV)) 
    #-------------------------# 
    #--- find first cpt ------#
    cpt=-1
    for t in range(0,lenc):
        if(Re[t]==-1): cpt = t; break
    if(cpt==-1 or PE_cr['errV'] <= errV): return (PE_vc, PE_vs) # if it cannot find better-trans, just ignore
    #-------------------------# 
    # update params {PE_vc : PE_vs}
    #tl.eprint("tm_st:%d, tm_ed:%d, cpt:%d"%(PE_vc["tm_st"], PE_vc["tm_ed"], cpt))
    PE_vc['tm_ed']=PE_vc['tm_st']+cpt
    PE_vs={'md':mdb.MD[Re[cpt+1]]['medoid'], 'cid':Re[cpt+1], 'Vc':Ve[cpt+1:,], 'Sc':Se[cpt+1:,], 'siset':{}, 'tm_st':PE_vc['tm_st']+(cpt+1), 'tm_ed':PE_cr['tm_ed'], 'errV':errV}
    #tl.eprint("tm_st:tm_ed: %d %d %d %d"%(PE_vc['tm_st'], PE_vc['tm_ed'], PE_vs['tm_st'], PE_vs['tm_ed']))
    # compute errorV 
    PE_vc['errV']=tl.RMSE(Xc[0:cpt], Ve[0:cpt])
    PE_vs['errV']=tl.RMSE(Xc[(cpt+1):lenc], Ve[(cpt+1):lenc])
    #tl.eprint("err_vc:%f - err_vs:%f (rho:%f)"%(PE_vc['errV'], PE_vs['errV'], mdb.rho_RE))
    return (PE_vc, PE_vs)




#--------------------------------#
# O-estimator
#--------------------------------#
# tc (current time point)
# Xorg (original data stream) 
# PE_vb (previous-previous pattern)
# PE_vp (previous pattern) 
# PE_vc (current pattern)
# mdb   (model parameter set) 
# Snaps (snapshots, etc.,... )
#--------------------------------#
def _O_estimator(tc, Xorg, PE_vb, PE_vp, PE_vc, mdb, Snaps):
    if(mdb.CC is tl.NO): return (PE_vb, PE_vp, PE_vc, mdb, Snaps) # if, no causal-chain, then ignore 
    lstep=Snaps['lstep']
    pstep=Snaps['pstep']
    wd_level=Snaps['wd_level']
    lmin=min(LMAX, max(LMIN,int(np.ceil(lstep*  LM_R))))
    smin=min(LMAX, max(LMIN,int(np.ceil(lstep*SMIN_R)))) 
    lmax=LMAX 
    PE_vc_prev=tl.dcopy(PE_vc) # current-best-previous
    PE_vc['tm_ed']=tc    # currrent time tick
    if(PE_vc['tm_ed']-PE_vc['tm_st']<lmin): return (PE_vb, PE_vp, PE_vc, mdb, Snaps)
    # create current window Xc
    Xc=Xorg[PE_vc['tm_st']:PE_vc['tm_ed'],:]
    if(COMMENT): tl.msg("%s |--- O_estimator: (wd=%d) t=%d:%d:%d:%d (rho:%f)-----------------|"%(IDT_E, wd_level, PE_vp['tm_st'], PE_vp['tm_ed'], PE_vc['tm_st'], PE_vc['tm_ed'],mdb.rho_RE))
   
    
    #-----------------------------------------------#
    # (I) find good-fit regime
    #-----------------------------------------------#
    #-----------------------------#
    # (A) regime update
    #-----------------------------#
    PE_RU = _RU(Xc, PE_vc, 'FIX')  # fixed, simply extend seg
    PE_vc = PE_RU
    if(PE_vc['errV'] > mdb.rho_RE): # if fixed is not good-enough 
        PE_RU = _RU(Xc, PE_vc, 'OPT') # update init value 
        if(PE_RU['errV'] <= PE_vc['errV']):
            PE_vc=PE_RU
    #-----------------------------#
    # (B) regime-trans update (if, current is known but not good-fit)
    #-----------------------------#
    PE_vs=[]
    if(PE_vc['cid'] !=-1 and PE_vc['errV'] > mdb.rho_RE): 
        (PE_RT_vc, PE_RT_vs) = _RT(Xc, mdb, smin, PE_vc) 
        if( PE_RT_vc['errV']< mdb.rho_RE and PE_RT_vs['errV']<mdb.rho_RE ):
            PE_vc=PE_RT_vc; PE_vs=PE_RT_vs
    #-----------------------------#
    # (C) regime reader, if update is not good-enough
    #-----------------------------#
    if(PE_vc['cid'] ==-1 or (PE_vc['errV'] > mdb.rho_RE) ): 
        PE_RR = _RR(Xc, mdb, PE_vc)
        if(PE_RR['errV'] <= PE_vc['errV']):
            PE_vc=PE_RR
    #-----------------------------#
    # (D) regime creation (estimator), only if the previous-fit is unknown (i.e., splitted) and current regime is not in MDB
    #-----------------------------#
    if(PE_vc_prev['cid']==-1 and PE_vc['errV'] > mdb.rho_RE):
        PE_RE = _RE(Xc, PE_vc['tm_st'], PE_vc['tm_ed'])
        if(PE_RE['errV'] <= PE_vc['errV']):
            PE_vc=PE_RE
    #-----------------------------------------------#
    # (II) split regime, if there is a new cut-point
    #-----------------------------------------------#
    #-----------------------------#
    lenc=PE_vc['tm_ed']-PE_vc['tm_st'] 
    #-----------------------------#
    # split segment now, if required
    #-----------------------------#
    if( (PE_vc['errV'] > mdb.rho_RE and lenc>lmin) or  # if current best is not-fit seg
        (PE_vs != [] and lenc>lmin) or # if any transisions
        (lenc > lmax) or  # if too long segment
        (PE_vc['tm_ed']+pstep >= Snaps['n']) ): # if it's close to the end point
    #-----------------------------#
        PE_vc=PE_vc_prev # use current-best-previous    
        #if(DBG0): tl.eprint("%s |-------- split segment: t=%d:%d, errV=%f (rho=%f)"%(IDT_E, PE_vc['tm_st'], PE_vc['tm_ed'], PE_vc['errV'], mdb.rho_RE))
        #----------------------------------------------------#
        # (d-a) if, good-enough-fit, then insert it into mdb
        if(PE_vc['errV'] <= mdb.rho_RE*RHO_MIN_R): 
            # (d-a-1) insert Xc into mdb
            (mdb, PE_vc) = mdb.update_md_insert(PE_vc) 
            # (d-a-2) insert MS(PE_vp, PE_vc) into mdb, if, we have PE_vp, PE_vc 
            if( (not _isNullP(PE_vp)) and (not _isNullP(PE_vc)) ):
                # (d-a-2-i) find opt-cut-point (+-pstep, argmin ||Vp-Xp||+||Vc-Xc||)
                (PE_vp, PE_vc) = _find_opt_cut(Xorg, PE_vp, PE_vc, lstep, lmin) #pstep)        
                # (d-a-2-ii) insert new regime-shift-dynamics into mdb
                mdb = mdb.update_ms_insert(PE_vp, PE_vc)  # add (from to)
                # (d-a-2-iii) update estimated events
                # previous set
                Snaps['Ve_full'][PE_vp['tm_st']:PE_vp['tm_ed'],:]=PE_vp['Vc'] 
                #Snaps['Se_full'][PE_vp['tm_st']:PE_vp['tm_ed'],:PE_vp['md'].k]=PE_vp['Sc'] 
                Snaps['Re_full'][PE_vp['tm_st']:PE_vp['tm_ed']]=PE_vp['cid'] 
                Snaps['Re_full'][PE_vp['tm_st']]=REID_CPT # cut-point (regime-switch)
                # current set
                Snaps['Ve_full'][PE_vc['tm_st']:PE_vc['tm_ed'],:]=PE_vc['Vc'] 
                #Snaps['Se_full'][PE_vc['tm_st']:PE_vc['tm_ed'],:PE_vc['md'].k]=PE_vc['Sc'] 
                Snaps['Re_full'][PE_vc['tm_st']:PE_vc['tm_ed']]=PE_vc['cid'] 
                Snaps['Re_full'][PE_vc['tm_st']]=REID_CPT # cut-point (regime-switch)
            #if(DBG0): tl.eprint("%s |-------- split segment(opt-cut): t=%d:%d, errV=%f (rho=%f)"%(IDT_E, PE_vc['tm_st'], PE_vc['tm_ed'], PE_vc['errV'], mdb.rho_RE))
        #----------------------------------------------------#
        # (d-b) reset params
        PE_vb=PE_vp; PE_vp=PE_vc; PE_vc=_initP(); 
        if(PE_vs!=[]): PE_vc=PE_vs
        PE_vc['tm_st']=PE_vp['tm_ed'];
    #-----------------------------#
        
    # return current results
    return (PE_vb, PE_vp, PE_vc, mdb, Snaps)



#--------------------------------#
# _find_opt_cut 
#--------------------------------#
# (find optimum cut-point)
# Vp[tm_st:tm_ed] vs. Vc[tm_st:tm_ed]
#--------------------------------#
def _find_opt_cut(Xorg, PE_vp, PE_vc, lenw, lmin):
    tp_st=PE_vp['tm_st']; tc_ed=PE_vc['tm_ed']; # start/end points
    tcut=PE_vp['tm_ed'] # current cut-point 
    errs=[]; locs=[]
    # find best-cut-position
    t_st=max(tp_st+lmin, tcut-int(lenw/2)); t_ed=min(tc_ed-lmin, tcut+int(lenw/2))
    PE_vps=[]; PE_vcs=[]
    if(t_st==t_ed): return (PE_vp, PE_vc)
    for loc in range(t_st, t_ed, int(max(1,(t_ed-t_st)/OPTCNT))): 
        PE_vpi=tl.dcopy(PE_vp); PE_vci=tl.dcopy(PE_vc)
        PE_vpi['tm_ed']=loc; PE_vci['tm_st']=loc;
        PE_vpi = _RU(Xorg[tp_st:loc,:], PE_vpi, 'OPT') 
        PE_vci = _RU(Xorg[loc:tc_ed,:], PE_vci, 'OPT') 
        err = PE_vpi['errV']+PE_vci['errV']
        PE_vps.append(PE_vpi); PE_vcs.append(PE_vci)
        locs.append(loc); errs.append(err)
        if(DBG0): tl.msg("%s ...... _find_opt_cut l:%d (%f + %f = %f)"%(IDT_E, loc, PE_vpi['errV'], PE_vci['errV'], err)) 
    ibest=np.argmin(errs); locbest=locs[ibest]
    if(DBG0): tl.msg("%s ...... _find_opt_cut l:%d (%f + %f = %f)"%(IDT_E, locbest, PE_vps[ibest]['errV'], PE_vcs[ibest]['errV'], err)) 
    PE_vp=PE_vps[ibest]; PE_vc=PE_vcs[ibest]
    return (PE_vp, PE_vc)




#--------------------------------#
# _FS, _FM, _FP
#--------------------------------#
# _FS: forecast using single regime 
# _FM: forecast using multi-step regimes
# _FP: forecast using current/last value Xc[tm_ed]
#--------------------------------#
# mdb (full model set)
# Xc (current window)
# lene (forecast length)
# lstep (lstep-ahead-forecast)
# PE (current parameter set)
# fixOropt (using fix/opt params) 
#--------------------------------#
#
#------------------------------------------------#
# forecast single
def _FS(mdb, Xc, lene, PE, fixORopt):
    md_c=PE['md'] #; si=md_c.si 
    (Se,Ve)=md_c.gen(lene)  #(Se,Ve)=md_c.forward(si, lene)
    Re=REID_NULL*np.ones(lene)
    errV=tl.RMSE(Xc, Ve[0:len(Xc)])
    if(_isUnfit(Ve, mdb)):
        Ve[:]=np.nan; Se[:]=np.nan; Re[:]=REID_NULL; errV=tl.INF
    if(DBG0): tl.msg("%s ++++++ _FS(%s): errV:%f (cid:%d) "%(IDT_F, fixORopt, errV, REID_NULL))
    PF={'Ve':Ve, 'Se':Se, 'Re':Re, 'errV':errV}
    return PF
#------------------------------------------------#
# forecast multi-step
def _FM(mdb, Xc, lene, smin, PE, fixORopt):
    # dictMS (see, om_trans.py)
    dictMS={'cid':PE['cid'], 'si':tl.dcopy(PE['md'].si), 'rho':mdb.rho_MS, 'smin':smin, 'errV':tl.INF} 
    if(fixORopt=='OPT'):
        dictMS = om_trans.fit_forward(Xc, mdb, dictMS, W_FM) #, DPS) 
    (Ve, Se, Re)=om_trans.gen_forward_multi_trans(Xc, mdb, dictMS, lene, MULTI_TRANS)
    #(Ve, Se, Re)=om_trans.gen_forward(mdb, dictMS, lene)
    errV=tl.RMSE(Xc, Ve[0:len(Xc)])
    if(_isUnfit(Ve, mdb)):
        Ve[:]=np.nan; Se[:]=np.nan; Re[:]=REID_NULL; errV=tl.INF
    if(DBG0): tl.msg("%s ++++++ _FM(%s): errV:%f (cid:%d) "%(IDT_F, fixORopt, errV, dictMS['cid']))
    PF={'Ve':Ve, 'Se':Se, 'Re':Re, 'errV':errV}
    return PF 
#------------------------------------------------#
# forecast cut&paste i.e., use current/latest event Xc[tm_ed]  
def _FP(Xc, lene):
    (lenc,d)=np.shape(Xc)
    Ve=np.zeros((lene,d))
    if(CUTNP=="MEAN"): Ve[0:lene,:]=tl.mynanmean(Xc, axis=0)
    if(CUTNP=="LAST"): Ve[0:lene,:]=Xc[-1,:] 
    Se=-1*np.ones((lene,1))
    Re=REID_NULL*np.ones(lene)
    errV=tl.RMSE(Xc, Ve[0:len(Xc)]) #=tl.INF #
    if(DBG0): tl.msg("%s ++++++ _FP(FIX): errV:inf (cid:%d) "%(IDT_F, REID_NULL))
    PF={'Ve':Ve, 'Se':Se, 'Re':Re, 'errV':errV, 'MSs':[]}
    return PF 
#------------------------------------------------#
#join a sequence of arrays along an existing axis.
def _con(A,B):
    return np.concatenate((A,B), axis=0)
# insert null values of length lene
def _padding(PF, lene):
    (lenc,d)=np.shape(PF['Ve'])
    (lenc,k)=np.shape(PF['Se'])
    PF['Ve']=_con( np.nan*np.zeros((lene,d)), PF['Ve'] )
    PF['Se']=_con( np.nan*np.zeros((lene,k)), PF['Se'] )
    PF['Re']=_con( np.nan*np.zeros((lene)),  PF['Re']  )
    return PF
#------------------------------------------------#
#------------------------------------------------#
# find unfit pattern (optional)
def _isUnfit(Ve, mdb):
    if(not WANT_AVOID_UNFIT): return False
    # check XminXmax (if, estimation (Ve) is too large or too small, ignore it 
    if( mdb.Xmin*AVOIDUNFIT_R > min(Ve.flatten()) 
            or mdb.Xmax*AVOIDUNFIT_R < max(Ve.flatten()) ) : return True
    return False
#------------------------------------------------#




#--------------------------------#
# O-generator
#--------------------------------#
# tc (current time point)
# Xorg (original data stream) 
# PE_vb (previous-previous pattern)
# PE_vp (previous pattern) 
# PE_vc (current pattern)
# mdb   (model parameter set) 
# Snaps (snapshots, etc.,... )
#--------------------------------#


#========================================================#
#
# ----------- PAST -------------|------ FUTURE ----------
#                               tc
# tb_st     tb_ed               |
# |         tp_st     tp_ed     |
# |         |         tc_st     tc_ed
# |         |         |         |         tf_st   tf_ed    
# |         |         |         |         |       |
# |---Vb----|---Vp----|---Vc----|---Vs----|--Vf---|
#
#========================================================#

def _O_generator(tc, Xorg, PE_vb, PE_vp, PE_vc, mdb, Snaps):
    #-----------------------------#
    # setting
    #-----------------------------#
    # (1) set timeticks    
    lstep=Snaps['lstep']; pstep=Snaps['pstep']; wd_level=Snaps['wd_level']
    smin=min(LMAX, max(LMIN,int(np.ceil(lstep*SMIN_R))))
    tb_st=PE_vb['tm_st']; tb_ed=PE_vb['tm_ed']
    tp_st=PE_vp['tm_st']; tp_ed=PE_vp['tm_ed']
    tc_st=PE_vc['tm_st']; tc_ed=PE_vc['tm_ed']
    # (2) if, previous P is null
    if(_isNullP(PE_vp)): 
        tb_st=PE_vc['tm_st']; tb_ed=PE_vc['tm_st']
        tp_st=PE_vc['tm_st']; tp_ed=PE_vc['tm_st']
    elif(_isNullP(PE_vb)): 
        tb_st=PE_vp['tm_st']; tb_ed=PE_vp['tm_st']
    if(_isNullP(PE_vc)): 
        tc_st=tp_ed; tc_ed=tp_ed 
    # (3) current time tick 
    tc_ed=tc 
    # (4) forecast window
    tf_st=tc_ed+lstep; tf_ed=min(tf_st+pstep,Snaps['n'])
    # (5) set original seq
    Xb=Xorg[tb_st:tb_ed,:]    # known
    Xbp=Xorg[tb_st:tp_ed,:]   # known
    Xbpc=Xorg[tb_st:tc_ed,:]  # known
    Xpc=Xorg[tp_st:tc_ed,:]   # known
    Xp=Xorg[tp_st:tp_ed,:]    # known
    Xc=Xorg[tc_st:tc_ed,:]    # known
    Xbpcsf=Xorg[tb_st:tf_ed,:]    # unknown
    Xpcsf=Xorg[tp_st:tf_ed,:]     # unknown
    Xcsf=Xorg[tc_st:tf_ed,:]      # unknown
    Xe=Xbpcsf
    # (6) seq-length 
    lenb=len(Xb);
    lenbp=len(Xbp); lenpcsf=len(Xpcsf); lencsf=len(Xcsf)
    lenpc=len(Xpc); lenp=len(Xp); lenc=len(Xc); lenbpcsf=len(Xbpcsf); #lenec=lene-lenp 
    lenbpc=len(Xbpc)
    if(COMMENT): tl.msg("%s |>>> O_generator:(wd=%d) t=%d:%d|%d:%d|%d:%d|%d:%d >>>>>>>>>>>|"%(IDT_F, wd_level, tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed, tf_st, tf_ed))

    #-----------------------------#
    # forecast
    #-----------------------------#
    # (A) cut-n-paste
    PF_cb=_FP(Xpc, lenpcsf)
    PF_cb=_padding(PF_cb, lenb)   
    
    if(mdb.CC is tl.YES):    
        #-------------------------------------------------#
        # (B-Vb) if pre-previous (Vb) is given, and known 
        if(USE_Vb): # if you want hop-step-jump-fit 
            if(PE_vb['cid']!=-1): 
                # (B-Vb-1) use current params
                PF_fix=_FM(mdb, Xbpc, lenbpcsf, smin, PE_vb, 'FIX')
                if(PF_fix['errV']<PF_cb['errV']):
                    PF_cb=PF_fix
                # (B-Vb-2) update params (if current params is not good enough)
                if(F_WANT_OPT and PF_cb['errV']>mdb.rho_RF):
                    PF_upd=_FM(mdb, Xbpc, lenbpcsf, smin, PE_vb, 'OPT')
                    if(PF_upd['errV'] < PF_cb['errV']):
                        PF_cb=PF_upd
        #-------------------------------------------------#
        # (B-Vp) if previous (Vp) is given, and known 
        if(PE_vp['cid']!=-1):    
            # (B-Vp-1) use current params
            PF_fix=_FM(mdb, Xpc, lenpcsf, smin, PE_vp, 'FIX')
            if(PF_fix['errV']<PF_cb['errV']):
                PF_cb=PF_fix
                PF_cb=_padding(PF_cb, lenb)   
            # (B-Vp-2) update params (if current params is not good enough)
            if(F_WANT_OPT and PF_cb['errV']>mdb.rho_RF):
                PF_upd=_FM(mdb, Xpc, lenpcsf, smin, PE_vp, 'OPT')
                if(PF_upd['errV'] < PF_cb['errV']):
                    PF_cb=PF_upd
                    PF_cb=_padding(PF_cb, lenb)   
        #-------------------------------------------------#
        # (C) if current best is BAD-fit, try Vc 
        if(PF_cb['errV']>mdb.rho_RF): 
            #-------------------------------------------------#
            # (C-1) if current is given, and known
            if(PE_vc['cid']!=-1):
                #-------------------------------------------------#
                # (C-1-1) use current params
                PF_fix=_FM(mdb, Xc, lencsf, smin, PE_vc, 'FIX') 
                if(PF_fix['errV']<PF_cb['errV']):
                    PF_cb=PF_fix
                    PF_cb=_padding(PF_cb, lenbp)   
                #-------------------------------------------------#
                # (C-1-2) update params (if current params is not good enough)
                if(F_WANT_OPT and PF_fix['errV']>mdb.rho_RF):
                    PF_upd=_FM(mdb, Xc, lencsf, smin, PE_vc, 'OPT') 
                    if(PF_upd['errV'] < PF_cb['errV']):
                        PF_cb=PF_upd
                        PF_cb=_padding(PF_cb, lenbp)   
            #-------------------------------------------------#
            # (C-2) if current is given, but unknown, try Vc but use only singlestep
            if( not _isNullP(PE_vc) and PE_vc['cid']==-1 ):
                # (C-2-1) use current param
                PF_fix=_FS(mdb, Xc, lencsf, PE_vc, 'FIX')
                if(PF_fix['errV']<PF_cb['errV']):
                    PF_cb=PF_fix
                    PF_cb=_padding(PF_cb, lenbp)   
        #-------------------------------------------------#

    #Snaps['Sf_full'][tf_st:tf_ed,:PF_cb['md'].k]=PF_cb['Se'][lenbpc+lstep:,:]
    Snaps['Vf_full'][tf_st:tf_ed,:]=PF_cb['Ve'][lenbpc+lstep:,:]
    Snaps['Rf_full'][tf_st:tf_ed]=PF_cb['Re'][lenbpc+lstep:]
    Snaps['Xe'].append(Xe)
    Snaps['Ve'].append(PF_cb['Ve']) 
    Snaps['Se'].append(PF_cb['Se']) 
    Snaps['Re'].append(PF_cb['Re']) 
    Snaps['Te'].append([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed, tf_st, tf_ed]) 
    
    return (Snaps)



#--------------------------------#
#     hyper-parameter setting    #
#--------------------------------#
def _compute_rho_min(Xorg, lstep): 
    (n,d)=np.shape(Xorg)
    lmin=max(LMIN, int(lstep*AVGR))
    errs=[]
    for i in range(0,AVG_TRIAL):
        tm_st=int((n-lmin)*np.random.rand()); tm_ed=tm_st+lmin
        if(tm_st<0 or tm_ed>=n): continue
        Xc=Xorg[tm_st:tm_ed,:]
        md_c=nl.NLDS(Xc, "t-%d"%(i))
        md_c=md_c.fit(W_RE)  # lmfit
        (Sc,Vc)=md_c.gen()   # generate events 
        err=tl.RMSE(Xc,Vc) 
        errs.append(err)
        if(DBG1): tl.msg("trial: %d, st:%d, ed:%d, err:%.2f"%(i, tm_st, tm_ed, err))
        #md_c.plot("%s_s_%d"%(mdb.fn,i)) 
    rho=om_mdb.compute_RHO_W_ERRs(errs)
    tl.eprint("estimated initial rho:  ", rho) #, errs)    
    return rho
#--------------------------------#




#---------------#
#     main      #
#---------------#
if __name__ == "__main__":

    tl.msg("OrbitMap")
    

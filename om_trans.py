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
import om_mdb as om_mdb
import om_trans_fit
DBG=tl.NO 
from sklearn import linear_model

LST_MINS=2 # last-switch
L_R=1.2 # generate Ve + extra
W_FM=om_mdb.W_FM
# regime switch
DBG_CPT=tl.YES # if, yes, switch_id=-1
# regimeID settings
REID_NULL=om_mdb.REID_NULL #regimeID (null) ... i.e., unknown regime
REID_CPT=om_mdb.REID_CPT   #regimeID (cpt)  ... i.e., cut-point (regime-switch)
#
#--------------------------------#
# dictMS={'cid': cid, 'si': si, 'rho': rho, 'smin': smin, 'err': errV}
#--------------------------------#

def gen_forward(mdb, dictMS, lene, dps=1):
    if(DBG): tl.msg("om_trans: generate (forward)")
    lene_l=int(lene*L_R) # generate Ve + extra
    #--------------------------------#
    MULTI_TRANS=tl.NO
    (Ve, Se, Re, Final) = _forward(mdb, dictMS, lene_l, dps, MULTI_TRANS)
    #--------------------------------#
    Ve=Ve[:lene,:]; Se=Se[:lene,:]; Re=Re[:lene];
    return (Ve, Se, Re)
    #--------------------------------#

def gen_forward_multi_trans(Xc, mdb, dictMS, lene, MULTI_TRANS=tl.YES, dps=1): 
    if(DBG): tl.msg("om_trans: generate (forward + multi_trans)")
    lene_l=int(lene*L_R) # generate Ve + extra
    #--------------------------------#
    (Ve, Se, Re, Final) = _forward(mdb, dictMS, lene_l, dps, MULTI_TRANS)
    if(MULTI_TRANS): (Ve) = _combine_multi_trans(Xc, Final, Ve, Re)
    #--------------------------------#
    Ve=Ve[:lene,:]; Se=Se[:lene,:]; Re=Re[:lene];
    return (Ve, Se, Re)
    #--------------------------------#



def fit_forward(Xe, mdb, dictMS, wtype='uniform', dps=1): #dps: current version, dps=1, only
    ftype='forward'
    dictMS = om_trans_fit.fit(Xe, mdb, ftype, dictMS, wtype, dps)
    return dictMS



def _combine_multi_trans(Xc, Stack, Ve, Re): 
    (lenc, d) = np.shape(Xc)
    (lene, k) = np.shape(Ve)
    r = len(Stack)
    tl.eprint("combine_multi_trans: (size: d:%d, lenc:%d, #candis:%d)"%(d, lenc, r))
    
    Ve_r=np.zeros((r,lene,d))
    Re_r=np.zeros((r,lene))
    for i in range(0, r): 
        Ve_r[i]=Stack[i]['Ve'][0:lene]
        Re_r[i]=Stack[i]['Re'][0:lene]
    Vc_r=Ve_r[:,0:lenc,:]

    Wt=_decompWt(Xc, Vc_r, r, d, lenc)
    Ve=np.zeros((lene,d))
    Ve=_mix_Ve_r(Ve_r, Ve, r, Wt)

    #---  DBG -------------#
    if(tl.NO): 
        for i in range(0,r):
            tl.plt.clf()
            tl.plt.subplot(211)
            tl.plt.plot(Ve_r[i])
            tl.plt.plot(Xc, '--')
            tl.plt.subplot(212)
            tl.plt.plot(Re_r[i])
            tl.plt.title(Wt)
            tl.savefig("./tmp/lenc_%d_lene_%d_stack_id_%d"%(lenc,lene, i), 'pdf')
    if(tl.NO): 
        tl.plt.clf()
        tl.plt.subplot(211)
        tl.plt.plot(Ve)
        tl.plt.plot(Xc, '--')
        tl.plt.subplot(212)
        #tl.plt.plot(Rem)
        tl.plt.plot(Re_r.T)
        tl.plt.title(Wt)
        tl.savefig("./tmp/lenc_%d_lene_%d_stack_full"%(lenc,lene), 'pdf')
    #---  DBG -------------#

    return (Ve)


#------------------------------------------------#
def _decompWt(Xc, Vc_r, r, d, lenc):
    #------------------------------------------------#
    # Given: 
    # Xc       (lenc x d)
    # Vc_r (r x lenc x d)
    # r: # of regimes
    # d: dimension
    # lenc: length of current window Xc
    #------------------------------------------------#
    # Out: 
    # Wt = {w[i]} = argmin ||Xc - sum(w[i]*Vc_r[i])|| (i=1,...r)
    #------------------------------------------------#
    # (1) find w[i] ... argmin(sum_{i}^{r} || Xc - w[i]*Vc_r[i] ||
    # (1-i) create weighted vector Wv
    Wv=tl.func_Weight(lenc,W_FM)
    #------------------------------------------------#
    # (1-ii) create flat vectors (Xc_flat/Vc_flat)
    Vc_r_flat=np.zeros((lenc*d, r)) # Vc_r (flatten), lenc*d x r 
    Xc_flat=np.zeros((lenc*d, 1))   # Xc   (flatten)  lenc*d x 1
    # (a) Vc_r_flat
    for i in range(0,r):
        for j in range(0,lenc):
            for jj in range(0,d):
                Vc_r_flat[j*d+jj][i]=Vc_r[i][j][jj]*Wv[j]
    # (b) Xc_flat
    for j in range(0,lenc):
        for jj in range(0,d):
            Xc_flat[j*d+jj][0]=Xc[j][jj]*Wv[j]
    #------------------------------------------------#
    # (2) find Wt={w[i]} (i=1,...r) Wt= argmin(||Xc - sum_{i}^{r} w[i]*Vc_r[i] ||
    #------------------------------------------------#
    # (a) linear-model (lasso)
    if(tl.YES):
        clf = linear_model.Lasso(alpha=0.01)
        clf.fit(Vc_r_flat, Xc_flat)
        Wt=clf.coef_
    # (b) linear-model (linear-regression)
    else:
        clf = linear_model.LinearRegression()
        clf.fit(Vc_r_flat, Xc_flat)
        Wt=clf.coef_[0]

    return Wt
#------------------------------------------------#

#------------------------------------------------#
def _mix_Ve_r(Ve_r, Ve, r, Wt):
    # compute mixed estimated events Ve = Wt*Ve_r 
    Ve[:]=0.0 
    for i in range(0,r):
        Ve+=Wt[i]*Ve_r[i]
    return Ve
#------------------------------------------------#





#--------------------=#
# 
#--------------------=#
def _get_trans_candidates(Vs, mdb, idx_c, smin, rho, MULTI_TRANS):
    candi=[] # MS candidates (i.e., errors, trans_time, MS, idxs(from_to))
    #--------------------------------------#
    for i in mdb.MD: 
        idxs="%d_%d"%(idx_c,i)
        if(idxs in mdb.MS): 
            msset=mdb.MS[idxs]
            #--------------------------------------#
            for ii in range(0, len(msset)):
                msset_ii=tl.dcopy(msset[ii]['medoid']) 
                if(msset_ii['active']==tl.NO): continue # if, this ms is not active, then, just ignore..
                # here, ms_rcd={idx_fr, idx_to, cpt, active, md_fr, vn_fr, md_to, si_to}
                msset_ii['idx_fr']=idx_c; msset_ii['idx_to']=i; 
                (t_best_ii, err_ii) = _find_nearest_t(Vs, msset_ii['vn_fr'], smin)
                #if(DBG): tl.eprint("%s, %d, %d, err:%f, (rho=%f)"%(idxs, ii, t_best_ii, err_ii,rho))
                if(err_ii <= rho): candi.append({'err':err_ii, 'cpt':t_best_ii, 'ms':msset_ii, 'idx':idxs, 'ii':ii}) 
            #--------------------------------------#
    # remove redundant candidates
    candi=_remove_redundancy(candi,smin, MULTI_TRANS)
    #for i in range(len(candi)): tl.eprint("%s, %d, %f"%(candi[i]['idx'], candi[i]['cpt'], candi[i]['err']))
    return candi
#--------------------=#
# remove redundant candidates in candi 
#--------------------=#
def _remove_redundancy(candi, smin, MULTI_TRANS):
    #----------------------#
    if(candi==[]): return candi # if no any candidates, return [] 
    #----------------------#
    lmin=smin # minimum length of over-lapping candidates 
    #----------------------#
    # (1) sort candidate
    errs=[]
    for i in range(0,len(candi)): errs.append(candi[i]['err'])
    #idx_best=np.argmin(errs) # find best id
    idx_sort=np.argsort(errs) # sorted id-list
    candi_new=tl.dcopy(candi)
    for i in range(0,len(idx_sort)):
        candi_new[i]=candi[idx_sort[i]]
    candi=candi_new; candi_new=[]
    #----------------------#
    # (2) if best-fit only, then return best candidate
    if(not MULTI_TRANS): candi_new.append(candi.pop(0)); return candi_new
    #----------------------#
    # (3) check redundant candidates
    while(len(candi)>0):
        c_i=candi.pop(0)
        want_del=tl.NO
        for c_j in candi_new:
            # if any over-lapping candidates
            if( c_i['idx']==c_j['idx'] and abs(c_i['cpt']-c_j['cpt'])<=lmin ):
                if(c_i['err']>c_j['err']):want_del=tl.YES
        if(not want_del): candi_new.append(c_i)
    #----------------------#
    return candi_new
#--------------------=#
# find cut-point it (smin <= it <= n-LST_MINS)
#--------------------=#
def _find_nearest_t(Vs, vec, smin):
    n=len(Vs)
    errs=tl.INF*np.ones((n))
    for it in range(smin, n-LST_MINS):
        errs[it]=tl.RMSE(Vs[it], vec)
    t_best=np.argmin(errs)
    err_best=errs[t_best]
    return (t_best, err_best)




def _create_Stc(t_c, idx_c, md_c, si, Se, Ve, Re):
    Stc={'t_c':t_c, 'idx_c':idx_c, 'md_c':md_c, 'si':si, 'Se':Se, 'Ve':Ve, 'Re':Re}
    return Stc
def _get_params_Stc(Stc):
    return (Stc['t_c'], Stc['idx_c'], Stc['md_c'], Stc['si'], Stc['Se'], Stc['Ve'], Stc['Re'])
def _get_params_misc(misc):
    return (misc['t_e'], misc['mdb'], misc['rho'], misc['smin'], misc['dps'])



def _gen_trans(Stack, Final, misc, MULTI_TRANS):
    # pop Stc in stack 
    Stc=Stack.pop(0)
    # get params from Stack
    (t_c, idx_c, md_c, si, Se, Ve, Re) = _get_params_Stc(Stc)
    # get params from misc setting
    (t_e, mdb, rho, smin, dps)= _get_params_misc(misc)
    #if(DBG): tl.eprint("t_c:%d, t_e:%d, idx_c:%d, smin:%d, rho:%f"%(t_c, t_e, idx_c, smin, rho))
    #--------------------------------------#
    # 0. set current regime variables
    #--------------------------------------#
    (Ss,Vs)=md_c.forward(si, (t_e-t_c+1), dps)
    Se[t_c:t_e,0:md_c.k]=Ss[0:t_e-t_c,:]
    Ve[t_c:t_e,:]=Vs[0:t_e-t_c,:]
    Re[t_c:t_e]=idx_c
    #--------------------------------------#
    Stc_c=_create_Stc(t_c, idx_c, md_c, si, Se, Ve, Re)
    #--------------------------------------#
    # (if it went to the end point (i.e., te-tc<smin) )
    if(t_e-t_c<smin): Final.append(Stc_c); return (Stack, Final) 
    #--------------------------------------#
    # 1. find candidate set : best-shift from md_c to ms_best
    #--------------------------------------#
    candi=_get_trans_candidates(Vs, mdb, idx_c, smin, rho, MULTI_TRANS) # MS candidates (i.e., err, ms, cpt, idx)
    #--------------------------------------#
    # (if no any close-enough vector, then return) 
    if(len(candi)==0): Final.append(Stc_c); return (Stack, Final) 
    #--------------------------------------#
    #--------------------------------------#
    # 2. for each candidate in candi, then shift it
    # (candi: if best-shift is good enough (<=rho) and length is long enough),
    #--------------------------------------#
    for i in range(0, len(candi)):
        Stc_new=tl.dcopy(Stc_c)
        (t_c, idx_c, md_c, si, Se, Ve, Re) = _get_params_Stc(Stc_new)
        # best candidate
        err_i = candi[i]['err']
        ms_i  = candi[i]['ms']
        cpt_i = t_c+candi[i]['cpt']
        #--------------------------------------#
        # 2-1. compute shifted-trajectory 
        #--------------------------------------#
        if(DBG): tl.eprint("|---@@@@@@@@@@@@@--- (cpt=%d, err=%f) %d >>> %d"%(cpt_i, err_i, ms_i['idx_fr'], ms_i['idx_to']))
        #--------------------------------------#
        if(DBG_CPT): Re[cpt_i-1]=REID_CPT
        idx_c = ms_i['idx_to']
        md_c  = ms_i['md_to']  # i.e., =mdb.MD[idx_c]['medoid'] 
        si    = ms_i['si_to']
        (s, o)=md_c.forward(si, 1, dps); 
        Se[cpt_i,0:md_c.k]=s; Ve[cpt_i,:]=o; Re[cpt_i]=idx_c;
        t_c=cpt_i+1; 
        #--------------------------------------#
        Stc_new=_create_Stc(t_c, idx_c, md_c, si, Se, Ve, Re)
        Stack.append(Stc_new) 
    #--------------------------------------#
    # return candidates (Stack and Final set)
    return (Stack, Final) 


#--------------------=#
# 0                 lene
# |----|------------|
#      t_c        t_e
#      --->
#
#--------------------=#
def _forward(mdb, dictMS, lene, dps, MULTI_TRANS):
    dps=1 # current setting: dps=1
    #--------------------------------------#
    # set current-regime md_c
    idx_c=dictMS['cid']
    si=dictMS['si']
    rho=dictMS['rho']
    smin=dictMS['smin']  
    #--------------------------------------#
    if(not idx_c in mdb.MD): tl.warning("_forward: cannot find %d"%(idx_c)); return ([],[],[])
    #--------------------------------------#
    md_c=mdb.MD[idx_c]['medoid']
    d=md_c.d; k=md_c.k;
    #--------------------------------------#
    Se=np.zeros((lene,md_c.kmax)) # hidden states
    Ve=np.zeros((lene,d)) # estimated events
    Re=np.zeros((lene))   # regime_id
    #--------------------------------------#
    #--------------------------------------#
    t_c=0; t_e=lene; 
    # set misc parameters
    misc={'t_e':t_e, 'mdb':mdb, 'rho':rho, 'smin':smin, 'dps':dps }
    # init Stack
    Stc=_create_Stc(t_c, idx_c, md_c, si, Se, Ve, Re)
    Final=[]; Stack=[]; Stack.append(Stc)
    # find all candidate set
    #--------------------------------------#
    while(len(Stack)>=1):
        (Stack, Final) = _gen_trans(Stack, Final, misc, MULTI_TRANS)
        if(DBG): tl.eprint("stack: %d"%len(Stack))
        if(DBG): tl.eprint("final: %d"%len(Final))
    #--------------------------------------#
    stc_best=[]
    #---  DBG -------------#
    if(tl.NO): 
        idx=0
        for f in Final:
            tl.plt.clf()
            tl.plt.subplot(211)
            tl.plt.plot(f['Ve'])
            tl.plt.subplot(212)
            tl.plt.plot(f['Re'])
            tl.savefig("./tmp/tc_%d_te_%d_cid_%d_idx_%d"%(t_c, t_e, idx_c, idx), 'pdf')
            idx+=1
    #---  DBG -------------#
    #--------------------------------------#
    Stc=Final[0]
    Ve=Stc['Ve']; Se=Stc['Se']; Re=Stc['Re']
    #--------------------------------------#
    return (Ve, Se, Re, Final)
    #--------------------------------------#






def _syn():
    DIR="../output/sanity_check/CG/synthetic_X3/"
    st=0; wd=60 #60; 
    ncast=60
    #st=548; wd=18 #20 #60; 
    smin=10 # smin=lstep
    cid=2 # if, est
    return (DIR, st, wd, ncast, smin, cid) 

def _mocap_c():
    DIR="../output/sanity_check/CG/mocap_c_21_21/"
    st=0; wd=400; cid=2 # if, est
    #st=200; wd=400; cid=1 # if, est
    #st=20; wd=400; cid=2 # if, est
    #st=400; wd=400; cid=3 # if, est
    smin=50; ncast=wd*2 # smin=lstep
    return (DIR, st, wd, ncast, smin, cid) 

def _mocap_c2():
    DIR="../output/sanity_check/CG/mocap_c_21_20/"
    st=0; wd=400; cid=2 # if, est
    #st=200; wd=400; cid=1 # if, est
    #st=400; wd=400; cid=3 # if, est
    smin=50; ncast=wd*2 # smin=lstep
    return (DIR, st, wd, ncast, smin, cid) 


#---------------#
#     main      #
#---------------#
if __name__ == "__main__":

    tl.msg("om_trans")
    (DIR, st, wd, ncast, smin, cid) = _syn()
    #(DIR, st, wd, ncast, smin, cid) = _mocap_c()
    #(DIR, st, wd, ncast, smin, cid) = _mocap_c2()

    mdbfn="%sMDB.obj"%(DIR)
    seqfn="%sXorg.txt"%(DIR)
    OUTDIR="%sout/"%(DIR)
    tl.mkdir(OUTDIR)
    tl.eprint(OUTDIR)
    mdb=tl.load_obj(mdbfn)
    #
    Xorg=tl.loadsq(seqfn).T
   

    
    #snapsfn="%sSnaps.mat"%(DIR)
    #Snaps=tl.load_mat(snapsfn)
    #print(len(mdb.MS))
    #tl.eprint(Snaps['Re_full'])
    #tl.eprint(np.shape(Snaps['Re_full']))
    #'''
    for k in mdb.MS:
        #print(mdb.MS)
        print(k)
        print(mdb.MS[k][0]['medoid']['cpt']) 
        print('# of ms', len(mdb.MS[k]))
        #print(mdb.MS[k][0]['medoid']['vn_fr'], mdb.MS[k][0]['medoid']['v0_to']) 
        tl.eprint('v0_fr', mdb.MS[k][0]['medoid']['v0_fr'], 'vn_fr', mdb.MS[k][0]['medoid']['vn_fr']) 
        tl.eprint('v0_to', mdb.MS[k][0]['medoid']['v0_to'], 'vn_to', mdb.MS[k][0]['medoid']['vn_to']) 
        tl.eprint('si_to', mdb.MS[k][0]['medoid']['si_to'])
    #'''

    
    rho=mdb.rho_MS #0.1 #1.0 #0.1 #1.0
    
    data=Xorg[st:st+wd,:]
    print(mdb.MD)
    si=mdb.MD[cid]['medoid'].si #"+0.2 #*(1+np.random.rand())
    lene=len(data) #200
    dictMS={'cid': cid, 'si': si, 'rho': rho, 'smin': smin}
    (Ve, Se, Re)=gen_forward(mdb, dictMS, lene)
    print(dictMS)

    tl.msg("plot original-est")
    tl.figure(1)
    tl.plt.subplot(221)
    tl.plt.plot(data, '--', color='grey')
    tl.plt.plot(Ve)
    tl.plt.title("rmse: %f"%(tl.RMSE(data, Ve)))
    tl.plt.ylim([np.min(data), np.max(data)])
    tl.plt.subplot(422)
    tl.plt.plot(tl.normalizeZ(Se))
    tl.plt.subplot(424)
    tl.plt.plot(Re)
    
    tl.eprint("========================================")

    tl.msg("model-switch-est")
    #dictMS['si']*=0.0; #mdb.MD[cid]['medoid'].si
    Xe=data
    dictMS = fit_forward(Xe, mdb, dictMS)
    print(dictMS)
    (Ve, Se, Re)=gen_forward(mdb, dictMS, ncast) #int(lene*ncast_r))
    print(dictMS)

    tl.save_txt(Xorg, '%sXorg'%(OUTDIR))
    tl.save_txt(Ve, '%sVe'%(OUTDIR))
    tl.save_txt(Re, '%sRe'%(OUTDIR))

    
    tl.plt.subplot(223)
    tl.plt.plot(Xorg[st:st+len(Ve),:], '--', color='lightgrey')
    tl.plt.plot(data, '--', color='grey')
    tl.plt.plot(Ve, '-')
    tl.plt.title("rmse: %f"%(tl.RMSE(data, Ve[0:lene,:])))
    tl.plt.xlim([0,len(Ve)])
    tl.plt.ylim([np.min(data), np.max(data)])
    tl.plt.subplot(4,4,11)
    tl.plt.plot(tl.normalizeZ(Se))
    tl.plt.subplot(4,4,12)
    tl.plt.plot(Ve[:,0],Ve[:,1], '-+')
    tl.plt.subplot(428)
    tl.plt.plot(Re)
    tl.plt.xlim([0,len(Ve)])
    
    
        
    # 
    tl.plt.show()
    tl.savefig('%s_tmp'%(OUTDIR),'pdf')






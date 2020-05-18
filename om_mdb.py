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
# om_mdb.py 
import tool as tl
import numpy as np
import nlds as nl
import graphviz as gv
DBG=tl.NO 
#-----------------------------#
REID_NULL=-3 #regimeID (null: [-1]) ... i.e., unknown regime
REID_CPT=-2  #regimeID (cpt: [-1])  ... i.e., cut-point (regime-switch)
#-----------------------------#
W_RR='uniform'  # weighted LMfit for RR (default: uniform or linear) 
W_RE='uniform'  # weighted LMfit for RE (default: uniform or linear)
W_FM='uniform'  # weighted LMfit for FM (default: uniform or linear)
# --- windows 
LMIN=5 #10 # minimum window length (default: 5) 
LMAX=500   # maximum window length (default: 500)
#-----------------------------#


class MDB:
    def __init__(self, lbl):
        if(DBG): tl.msg("MDB")
        self.lbl=lbl # label
        #  model params
        self.MD={}  # MD[c]  model-dynamics
        self.MS={}  # MS["from_to"]   model-shift
        self.cid_counter=0 # cid_counter
        # rho
        self.rho_RE=tl.INF #RHOini_RE
        #self.errVes=[]
        self.rho_RF=tl.INF #RHOini_RF
        #self.errVfs=[]
        self.rho_MS=tl.INF #RHOini_ms #RHOini # RHOini, notfin
        # causal-cast
        self.CC=tl.YES
        self.Xmin=   tl.INF
        self.Xmax= - tl.INF

    #---------------------#
    # get info 
    #---------------------#
    def get_c(self):
        return len(self.MD)
    def get_cnt_MD_obj(self):
        cnt=0
        for i in self.MD:
            cnt+=len(self.MD[i]['object'])
        return cnt
    #---------------------#
    # search mdb 
    #---------------------#
    def search_md(self, Xc, Si=[], dps=1):
        (Vc, md_best, Si, cid) = _search_md(self, Xc, Si, dps)
        if(DBG): tl.msg("|-------- MDB-MD: search_md dps:%d (id=%d / # of MD=%d)"%(dps, cid, self.get_c()))
        return (Vc, md_best, Si, cid)

    #---------------------#
    # update mdb 
    #---------------------#
    def update_rho(self, rho):
        self.rho_RE=rho; self.rho_RF=rho; self.rho_MS=rho
    
    def update_md_insert(self, PE_ins): #_cid, md_new):
        return _update_md_insert(self, PE_ins) #cid, md_new)
    def update_ms_insert(self, PE_fr, PE_to): 
        return _update_ms_insert(self, PE_fr, PE_to) 
    
    def update_apoptosis(self, Snaps, cmax_md, cmax_ms, REFINEMENT=tl.NO):
        return _update_apoptosis(self, Snaps, cmax_md, cmax_ms, REFINEMENT)

    #def update_ms_block(self, idx_fr, idx_to, cpt): # UNUSED
    #    return _block_ms(self, idx_fr, idx_to, cpt)
 
    def update_Xminmax(self, Xc): 
        return _update_Xminmax(self, Xc)
        
 

    def create_CGraph_rcds(self):
        CGraph = _create_CGraph_rcds(self)
        return CGraph

    def plot_CG(self, fn):
        _save_CGraph(self, "%s"%(fn))

    # init objects in MD and MS 
    def init_objects(self):
        for i in self.MD:
            self.MD[i]['object']=[]
        for i in self.MS:
            for j in range(0,len(self.MS[i])):
                self.MS[i][j]['object']=[]
        return self 


def compute_RHO_W_ERRs(errs): 
    return _compute_RHO_W_ERRs(errs)

#1.64: 90%, 2.0: 95.45
GAUSS_R=1.64 
def _gaussTH(X):
    v_mean=tl.mynanmean(X)
    v_std=np.nanstd(X)
    v_th=v_mean+GAUSS_R*v_std # lower-bound (e.g., -2*sigma : 95.45%)
    return (v_mean, v_std, v_th)

def _compute_RHO_W_ERRs(errs):
    (err_m, err_s, err_th)=_gaussTH(errs)
    rho=err_th
    if(DBG): tl.msg("compute RHO using errs : rho=%f (m:%f, s:%f)"%(err_th, err_m, err_s))
    return rho



# update Xmin, Xmax
def _update_Xminmax(mdb, Xc):
    mdb.Xmin=min( mdb.Xmin, min(Xc.flatten()) )
    mdb.Xmax=max( mdb.Xmax, max(Xc.flatten()) )
    #tl.eprint("Xmin: %f, Xmax: %f"%(mdb.Xmin, mdb.Xmax))
    return mdb


'''
def _update_rho_RE(mdb, err):
    mdb.errVes.extend(err)
    errs=mdb.errVes
    (err_m, err_s, err_th)=_gaussTH(errs)
    # notfinA
    #err_th=np.mean(errs)
    mdb.rho_RE=err_th; mdb.rho_MS=err_th
    mdb.rho_RF=err_th 
    if(DBG): tl.msg("update_RHO (RE) : %f (m:%f, s:%f)"%(err_th, err_m, err_s))
    return mdb
'''

'''
def _block_ms(mdb, idx_fr, idx_to, cpt):
    idxs="%d_%d"%(idx_fr, idx_to)
    if(idxs in mdb.MS):
        mss=mdb.MS[idxs]
        for ms in mss:
            ms_m=ms['medoid']
            if(ms_m['cpt']==cpt):
                ms_m['block']=tl.YES
                if(DBG): tl.msg("|-------- MDB-MS (block):   --- %d_%d, cpt:%d"%(idx_fr, idx_to, cpt))
    return mdb
'''


#----------------------------------------------#
#
# Search_md
#
#----------------------------------------------#
def _search_md(mdb, Xc, siset, dps): 
    if(mdb.get_c()==0): return ([], [], [], -1)  # no data
    #----------------------------------------------#
    siset_new={}
    err_best=tl.INF; cid_best=-1; md_best=[]
    # for each regime/medoid, try argmin_si(Xc-Vc)
    for i in mdb.MD:
        md_c=tl.dcopy(mdb.MD[i]['medoid'])
        md_c.data=Xc; md_c.n=np.size(Xc,0)
        if(i in siset): md_c.si= siset[i]['si'] # use previous si
        md_c=md_c.fit_si(W_RR, dps)
        err_c=md_c.rmse()
        siset_new[i]={'cid':i, 'md':md_c, 'si':md_c.si, 'err':err_c}
        if(err_best > err_c or err_best==tl.INF):
            err_best=err_c; md_best=md_c; cid_best=i;
    (Sc,Vc)=md_best.gen()
    #----------------------------------------------#
    return (Vc, md_best, siset_new, cid_best) 
#----------------------------------------------#
   

#----------------------------------------------#
#
# update_md
#
#----------------------------------------------#
def _update_md_insert(mdb, PE_ins): 
    cid=PE_ins['cid']; md_ins=PE_ins['md'];
    if(DBG): tl.msg("om_mdb: MDB-MD (insert): cid:%d (c=%d)"%(cid, mdb.get_c()))
    # (A) if, known regime, then insert & return  
    if(cid!=-1): 
        mdb.MD[cid]['object'].append(md_ins)
    # (B) if unknown regime, then create new MD
    if(cid==-1):
        mdb.cid_counter+=1; 
        cid=mdb.cid_counter;
        mdb.MD[cid]={'medoid':md_ins, 'object':[md_ins]}
        PE_ins['cid']=cid
    # (C) return result
    return (mdb, PE_ins)



'''
def _update_ms_insert(mdb, PE_fr, PE_to):
    idx_fr=PE_fr['cid']; idx_to=PE_to['cid'];
    # if not exist, just ignore
    if(idx_fr == -1 or idx_to == -1): return mdb 
    # create new ms record
    ms_new=_create_new_ms_rcd(PE_fr, PE_to)
    # (a) find fr_to in mdb
    idxs="%d_%d"%(idx_fr, idx_to)
    if(not (idxs in mdb.MS)): 
        mdb.MS[idxs]=[]
        mdb.MS[idxs].append({'medoid':ms_new, 'object':[ms_new]})
    return mdb
'''

def _update_ms_insert(mdb, PE_fr, PE_to):
    idx_fr=PE_fr['cid']; idx_to=PE_to['cid'];
    # if not exist, just ignore
    if(idx_fr == -1 or idx_to == -1): return mdb 
    # create new ms record
    ms_new=_create_new_ms_rcd(PE_fr, PE_to)
    # (a) find fr_to in mdb
    idxs="%d_%d"%(idx_fr, idx_to)
    if(not (idxs in mdb.MS)): mdb.MS[idxs]=[]
    # best-candidate  
    ms_best=-1; err_best=tl.INF
    # (a-i) find closest mss
    for ms_idxs_i in mdb.MS[idxs]:
        ms_d=ms_idxs_i['medoid']
        err_i=0.5*( tl.RMSE(ms_d['vn_fr'],ms_new['vn_fr']) + tl.RMSE(ms_d['v0_to'],ms_new['v0_to']) )
        #tl.eprint(mdb.rho_MS, err_i)
        #tl.eprint(ms_d['vn_fr'], ms_new['vn_fr'],  ms_d['v0_to'],ms_new['v0_to'])
        if(err_i<err_best):
            ms_best=ms_idxs_i; err_best=err_i;
    #tl.eprint("err_best: %f (%f)"%(err_best, mdb.rho_MS))
    # (b) update MS[idxs]
    if(err_best<mdb.rho_MS): # if we have similar ms 
        ms_best['object'].append(ms_new)
        if(DBG): tl.msg("MDB-MS (insert): # of object=%d"%(len(ms_best['object'])))
    else: # if unknown, then create new MS
        mdb.MS[idxs].append({'medoid':ms_new, 'object':[ms_new]})
    return mdb


def _create_new_ms_rcd(PE_fr, PE_to):
    idx_fr=PE_fr['cid']; md_fr=PE_fr['md']
    idx_to=PE_to['cid']; md_to=PE_to['md']
    cpt=PE_to['tm_st']
    # create new regime-shift record        
    # here, ms_rcd={cpt, active, md_fr, vn_fr, md_to, si_to}
    rcd={'cpt': cpt, 'active': tl.YES} 
    # --- from --- #
    (Sta, Obs) = md_fr.gen();
    rcd['md_fr']=md_fr;     
    rcd['si_fr']=Sta[0,:]; rcd['sn_fr']=Sta[-1,:]; 
    rcd['v0_fr']=Obs[0,:]; rcd['vn_fr']=Obs[-1,:]
    # --- to --- #
    (Sta, Obs) = md_to.gen();
    rcd['md_to']=md_to;
    rcd['si_to']=Sta[0,:]; rcd['sn_to']=Sta[-1,:]; 
    rcd['v0_to']=Obs[0,:]; rcd['vn_to']=Obs[-1,:]
    if(DBG): tl.msg("|-------- MDB-MS (object):  cpt: %d, from: %d, to:%d (%s, %s)"%(cpt, idx_fr, idx_to, md_fr.fn, md_to.fn))
    return rcd



#----------------------------------------------#
#
# update_apoptosis 
#
#----------------------------------------------#
def _update_apoptosis(mdb, Snaps, cmax_md, cmax_ms, REFINEMENT=tl.NO):
    if(REFINEMENT): mdb = _update_ms_apoptosis(mdb, tl.INF) # reset all MS if required (refinement)
    if(REFINEMENT): (mdb, Snaps) = _update_md_absorb(mdb, Snaps) # absorb MDs if required (refinement)
    (mdb, Snaps) = _update_md_apoptosis(mdb, Snaps, cmax_md)
    mdb = _update_ms_apoptosis(mdb, cmax_ms)
    (mdb, MD_IDs) = _update_md_ms_reset_cIDs(mdb) 
    Snaps = _init_MD_ID_Snaps(Snaps, MD_IDs)
    if(REFINEMENT): (mdb, Snaps) = _update_md_ms_medoid_refinement(mdb, Snaps) # refine each MD
    return (mdb, Snaps)



#--------------------------#
#
# refinement (update medoids)
#
#--------------------------#
def _update_md_ms_medoid_refinement(mdb, Snaps): 
    # for each md in MD 
    for i in mdb.MD:
        _md_medoid_refinement_each(mdb, Snaps, i)
    return (mdb, Snaps)
#--------------------------#
# MD cleansing and absorbing (find appropriate medoid in MDB, and delete useless md if any)
def _md_medoid_refinement_each(mdb, Snaps, cid_i): 
    dps=Snaps['DPS']
    Xorg=Snaps['Xorg']
    md_c=tl.dcopy(mdb.MD[cid_i]['medoid'])
    # (a) find longest segment
    (tm_st, tm_ed) = _find_longest_segment(Snaps, cid_i)
    if(tm_st==-1 and tm_ed==-1): return (mdb, Snaps)
    # set current window
    Xc=Xorg[tm_st:tm_ed,:]; len_Xc=tm_ed-tm_st
    # (b) try current model vs. new model
    # (b-1) current model
    md_c.data=Xc
    md_c=md_c.fit_si(W_RR, dps) # si-fit using current param
    (Sc,Vc)=md_c.gen(len_Xc); err_cr=tl.RMSE(Xc,Vc)
    # (b-2) new model (estimate param using Xc)
    md_new=nl.NLDS(Xc, "medoid-refinement-t:%d:%d"%(tm_st, tm_ed))
    md_new.data=Xc
    md_new=md_new.fit(W_RE)  # lmfit (regime creation)
    (Sc,Vc)=md_new.gen(len_Xc); err_new=tl.RMSE(Xc,Vc) 
    # (c) if it finds better model, then replace 
    if(err_new < err_cr): md_c=md_new
    return (mdb, Snaps)

#-------------------------------------------------#
def _find_longest_segment(Snaps, cid): 
    tm_st=-1; tm_ed=-1; len_max=-1;
    lmax=LMAX
    #len_seed_mx=min(lmin*LMIN_MAX_SEED_R, LMAX)
    seglist=_get_seglist(Snaps['Re_full'], cid)
    if(seglist=={}): 
        return (-1, -1) # not-found
    for i in seglist:
        seg=seglist[i]
        if(len_max < seg['tm_len']): 
            tm_st=seg['tm_st']; tm_ed=seg['tm_ed']; len_max=seg['tm_len']
    #tl.eprint("find_longest_seg: %d, %d, %d (before)"%(tm_st, tm_ed, len_max))
    if(len_max>lmax):
        tm_st=int(tm_st + (len_max-lmax)*np.random.rand() )
        tm_ed=tm_st+lmax
    #tl.eprint("find_longest_seg: %d, %d, %d (after)"%(tm_st, tm_ed, len_max))
    return (tm_st, tm_ed)
#-------------------------------------------------#
def _get_seglist(Re_full, cid):
    n=len(Re_full)
    seglist={}; tm_idx=0;
    tm_st=-1; tm_ed=-1; tm_len=-1;
    for i in range(0,n):
        if(Re_full[i]==cid): 
            if(tm_st==-1): tm_st=i; # start segment
        if(Re_full[i] != cid):
            if(tm_st !=-1): # end segment
                tm_ed=i; tm_len=tm_ed-tm_st+1;
                seglist[tm_idx]={'tm_st':tm_st, 'tm_ed':tm_ed, 'tm_len':tm_len}
                tm_st=-1; tm_ed=-1; tm_idx+=1;
    return seglist
#-------------------------------------------------#






#--------------------------#
#
# absorb (cleansing)
#
#--------------------------#
def _update_md_absorb(mdb, Snaps): 
    # for each md in  MD absorb
    for i in range(1, mdb.cid_counter+1): 
        if i in mdb.MD.keys(): 
            (mdb, Snaps) = _md_absorb_each(mdb, Snaps, i)
    return (mdb, Snaps)
#--------------------------#
# MD cleansing and absorbing (find appropriate medoid in MDB, and delete useless md if any)
def _md_absorb_each(mdb, Snaps, cid_i): 
    mdi=mdb.MD[cid_i]
    Xci=mdi['medoid'].data
    Si=[]; dps=Snaps['DPS']
    # find most similar MD in MDB (except current cid_i)
    del(mdb.MD[cid_i]) # remove cid_i(temporal)
    (Vc, md_best, Si, cid_j)=mdb.search_md(Xci, Si, dps)
    mdb.MD[cid_i]=mdi # add cid_i
    # if, cannot find, then, do nothing
    if(cid_j == -1): return (mdb, Snaps)
    if(DBG): tl.msg("om_mdb: md_absorb_each: cid_i:%d,cid_j:%d, (c=%d), errC:%f, rho:%f"%(cid_i,cid_j, mdb.get_c(),md_best.rmse(),mdb.rho_RE))
    # if, it finds 
    mdj=mdb.MD[cid_j]
    # copy model
    mdi_m=tl.dcopy(mdi['medoid'])
    mdj_m=tl.dcopy(mdj['medoid'])
    # switch data i,j 
    mdi_m.data=mdj['medoid'].data
    mdj_m.data=mdi['medoid'].data
    mdi_m.fit_si(W_RR,dps)
    mdj_m.fit_si(W_RR,dps)
    # compute errors (i vs j)
    err_i=mdi_m.rmse()
    err_j=mdj_m.rmse()
    if(DBG): tl.msg("om_mdb: switchErr: (%d,%d) erri:%f errj:%f"%(cid_i, cid_j, err_i, err_j))
    if(err_i<err_j and err_i<mdb.rho_RE):
        if(DBG): tl.msg("md_absorb: REPLACE(absorb) %d->%d"%(cid_j,cid_i))
        mdb.MD[cid_i]['object'].append(mdj) # add regime j (object) -> regime i
        mdb=_update_md_ms_delete_idx(mdb, cid_j) # delete regime j 
        MD_IDs={}; MD_IDs[cid_j]=cid_i; Snaps=_init_MD_ID_Snaps(Snaps, MD_IDs) # Snaps: replace j -> i 
    if(err_j<err_i and err_j<mdb.rho_RE):
        if(DBG): tl.msg("md_absorb: REPLACE(absorb) %d->%d"%(cid_i,cid_j))
        mdb.MD[cid_j]['object'].append(mdi) # add regime j (object) -> regime i
        mdb=_update_md_ms_delete_idx(mdb, cid_i) # delete regime i
        MD_IDs={}; MD_IDs[cid_i]=cid_j; Snaps=_init_MD_ID_Snaps(Snaps, MD_IDs) # Snaps: replace i -> j 
    return (mdb, Snaps)
#--------------------------#



#----------------------------------------------#
#
# reset regime ID 
#
#----------------------------------------------#
def _update_md_ms_reset_cIDs(mdb):  
    mdb_org=tl.dcopy(mdb)
    mdb.cid_counter=0 # new_mdb counter = 0,1,...
    # create new MD_IDs
    MD_IDs={}; counter=1
    for i in range(1,mdb_org.cid_counter+1):
        if(i in mdb_org.MD): 
            MD_IDs[i]=counter; counter+=1
    # replace ID (from oldID to newID)
    for i in MD_IDs:
        oldID=i
        newID=MD_IDs[i]
        # MD (node), replace from oldID to newID
        del(mdb.MD[oldID])
        mdb.MD[newID]=mdb_org.MD[oldID]
        mdb.cid_counter+=1
        for j in MD_IDs:
            oldID2=j
            newID2=MD_IDs[j]
            oldIdx="%d_%d"%(oldID, oldID2)
            newIdx="%d_%d"%(newID, newID2)
            # MS (edge)
            if(oldIdx in mdb_org.MS): 
                del(mdb.MS[oldIdx])
                mdb.MS[newIdx]=mdb_org.MS[oldIdx]
    return (mdb, MD_IDs)


#----------------------------------------------#
#
# init/replace regime ID in Snaps
#
#----------------------------------------------#
#-------------------------------#
def _init_MD_ID_Snaps(Snaps, MD_IDs):
    #tl.msg('init_MD_IDs'); tl.msg(MD_IDs)
    Snaps['Re_full']=_replace_IDs(Snaps['Re_full'], MD_IDs)
    Snaps['Rf_full']=_replace_IDs(Snaps['Rf_full'], MD_IDs)
    for tt in range(0,len(Snaps['Re'])):
        Snaps['Re'][tt]=_replace_IDs(Snaps['Re'][tt],MD_IDs)    
    return Snaps
#-------------------------------#
def _replace_IDs(Re, MD_IDs):
    for t in range(0,len(Re)):
        if(Re[t] in MD_IDs): Re[t]=MD_IDs[Re[t]]
        #else: Re[t]=REID_NULL
    return Re
#-------------------------------#



#-------------------------------#
#
# apoptosis
#
#-------------------------------#
# delete md if # of samples is less than minc
def _update_md_apoptosis(mdb, Snaps, minc):
    # (a) find id list
    rlst=[] # rls: remove list
    for i in mdb.MD: 
        md=mdb.MD[i]
        cnum=len(md['object'])
        if(DBG): tl.msg("|-------- MDB-MD (apoptosis-minc:%d):  %s, #ofsamples:%d"%(minc, i, cnum))
        if(cnum<=minc):
            rlst.append(i)
    # (b) remove id from mdb
    for i in rlst:
        mdb=_update_md_ms_delete_idx(mdb, i) # delete regime_id=i from MDB
        MD_IDs={}; MD_IDs[i]=REID_NULL; Snaps=_init_MD_ID_Snaps(Snaps, MD_IDs) # Snaps: delete regime_id=i
    return (mdb, Snaps) 

# delete ms if # of samples is less than minc
def _update_ms_apoptosis(mdb, minc):
    # (a) check each len(MS[idxs][j]) 
    for idxs in mdb.MS: 
        rlst=[] # rls: remove list
        for j in range(0,len(mdb.MS[idxs])):
            cnum=len(mdb.MS[idxs][j]['object'])
            #blocked=mdb.MS[idxs][j]['medoid']['block']
            active=mdb.MS[idxs][j]['medoid']['active']
            if(DBG): tl.msg("|-------- MDB-MS (apoptosis-minc:%d):  %s[%d], #ofsamples:%d, active=%d"%(minc, idxs, j, cnum, active))
            #if(cnum<=minc): #  or blocked): notfinA
            if(cnum<=minc  or (active == tl.NO)): # if less than minc or non-active
                rlst.append(j)
        # (a') remove id from mdb
        rlst.sort(reverse=True)
        for j in rlst:
            mdb.MS[idxs].pop(j)
    # (b) check each len(MS[idxs]) 
    rlst=[] # rls: remove list
    for idxs in mdb.MS: 
        cnum=len(mdb.MS[idxs])
        if(DBG): tl.msg("|-------- MDB-MS (apoptosis:%d):  %s[xx], #ofsamples:%d"%(minc, idxs, cnum))
        if(cnum<=0): # if is_empty
            rlst.append(idxs)
    # (b') remove id from mdb
    for idxs in rlst:
        del(mdb.MS[idxs])
    return mdb 

# delete MD,MS from index 
def _update_md_ms_delete_idx(mdb, idx):
    # (a) remove MS
    for j in mdb.MD:
        idxs="%d_%d"%(idx,j)
        if(idxs in mdb.MS):
            del(mdb.MS[idxs])
        idxs="%d_%d"%(j,idx)
        if(idxs in mdb.MS):
            del(mdb.MS[idxs])
    # (b) remove MD
    if(idx in mdb.MD): 
        del(mdb.MD[idx])
    return mdb     
    


#----------------------------------------------#
#
# save graph 
#
#----------------------------------------------#

def _save_CGraph(mdb, fn):
    CGraph = mdb.create_CGraph_rcds()
    gv.write_dot(CGraph, fn) #graphviz 
    #_plot_MS_objects(mdb, fn)  # debug

def _create_CGraph_rcds(mdb):
    md_object_cnt=mdb.get_cnt_MD_obj()
    rcds_MD=[]; rcds_MS=[]
    # node
    for i in mdb.MD: 
        md=mdb.MD[i]
        cnum=len(md['object'])
        size=max(1.0, min(10.0, 10.0*cnum/(md_object_cnt+tl.ZERO)))
        #rcd_MD={'id':i, 'lbl':'#%d (%d)'%(i, cnum), 'size':size} #notfinA
        rcd_MD={'id':i, 'cnt':cnum} #notfinA
        rcds_MD.append(rcd_MD)
    # edge
    for i in mdb.MD:
        for j in mdb.MD:
            idxs="%d_%d"%(i,j)
            if(idxs in mdb.MS):
                mss=mdb.MS[idxs]
                cnum=len(mss)
                cnum_sum=0
                for ms_i in mss: cnum_sum+=len(ms_i['object'])
                size=max(1.0, min(10.0, 100.0*cnum_sum/(md_object_cnt+tl.ZERO)))
                #rcd_MS={'fr':i, 'to':j, 'lbl':'%d'%(cnum_sum), 'size':size}
                rcd_MS={'fr':i, 'to':j, 'tcnt':cnum, 'cnt':cnum_sum}
                if(cnum>=1):rcds_MS.append(rcd_MS) 
    CGraph={'MD': rcds_MD, 'MS':rcds_MS}
    return CGraph


def _plot_MS_objects(mdb, fn):
    dbgmatM_fr=[]; dbgmatM_to=[]
    dbgmat_fr=[]; dbgmat_to=[]
    # edge
    for i in mdb.MD:
        for j in mdb.MD:
            idxs="%d_%d"%(i,j)
            if(idxs in mdb.MS):
                mss=mdb.MS[idxs]
                for ms_i in mss: 
                    # medoid
                    if(dbgmatM_fr==[]): dbgmatM_fr=[ms_i['medoid']['vn_fr']]
                    else: dbgmatM_fr = np.concatenate((dbgmatM_fr, [ms_i['medoid']['vn_fr']]), axis=0)
                    if(dbgmatM_to==[]): dbgmatM_to=[ms_i['medoid']['v0_to']]
                    else: dbgmatM_to = np.concatenate((dbgmatM_to, [ms_i['medoid']['v0_to']]), axis=0)
                    # objects
                    for iii in range(0, len(ms_i['object']) ):
                        if(dbgmat_fr==[]): dbgmat_fr=[ms_i['object'][iii]['vn_fr']]
                        else: dbgmat_fr = np.concatenate((dbgmat_fr, [ms_i['object'][iii]['vn_fr']]), axis=0)
                        if(dbgmat_to==[]): dbgmat_to=[ms_i['object'][iii]['v0_to']]
                        else: dbgmat_to = np.concatenate((dbgmat_to, [ms_i['object'][iii]['v0_to']]), axis=0)
    tl.save_txt(dbgmatM_fr, "%sMS_M_scatter_fr"%fn)
    tl.save_txt(dbgmatM_to, "%sMS_M_scatter_to"%fn)
    tl.save_txt(dbgmat_fr, "%sMS_scatter_fr"%fn)
    tl.save_txt(dbgmat_to, "%sMS_scatter_to"%fn)
    return (dbgmat_fr, dbgmat_to) 




#---------------#
#     main      #
#---------------#
if __name__ == "__main__":
    from om_mdb import MDB 
    fn="../output/tmp/MDBtest/"
    mdb=MDB(fn)
    print(mdb.get_c())
    #mdb.save_obj()
    
    mdbfn='../output/tmp/syn_X3/scan/_XH0/MDB.obj'
    mdbfn='../output/CG_CS/jma/lstep12_ms3_sr1_r8675/scan/_XH1/MDB.obj'
    mdb=tl.load_obj(mdbfn)
    print(mdb.get_c())
    mdb.plot_CG('%sCGraph'%(fn))

    

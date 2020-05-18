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
import graphviz as gv
import orbitmap as orbitmap
#------------------------------------------------#
DBG=tl.NO 
PLOT_R=tl.YES
ANIMATION=tl.NO # plot animation (default: NO)
FULLSAVE=tl.NO # full plot & save
MULTI=tl.YES # multiprocessing (default: YES)
MSCALE_H=0  # # of scales (see _set_env)
SMTH=tl.YES  # smoothing results
MINC=1 #2 # show result if c>=MINC
#------------------------------------------------#


#---------------------------------------------#
# convert (ImageMagick) : 
# /etc/ImageMagick-6/policy.xml
# [ before ]
# policy domain="coder" rights="none" pattern="PDF"
# -> 
# [after] 
# policy domain="coder" rights="read|write" pattern="PDF"
#---------------------------------------------#

#--------------------------------#
def run_viz(cdir, animation=ANIMATION):
    _set_env(cdir, animation)
    tl.msg("om_viz (visualization), animation=%d, mscale=%d"%(animation, MSCALE_H))
    if(MSCALE_H==0):_run_single(cdir, animation)
    else: _run_mscale(cdir, animation)
    #--------------------------------#
    tl.msg('om_viz end.')
    #--------------------------------#
#--------------------------------#

#--------------------------------#
def plotResultsEF(Snaps, mdb, outdir):
    _plotResultsEF(Snaps, mdb, outdir)

def plotResultsE(Snaps, mdb, outdir):
    _plotResultsE(Snaps, mdb, outdir)

def plotResultsF(Snaps, mdb, outdir):
    _plotResultsF(Snaps, mdb, outdir)

def saveResults_txt(Snaps, mdb, outdir):
    _saveResults_txt(Snaps, mdb, outdir)
#--------------------------------#

#--------------------------------#
def plotMD(mdb, outdir):
    for i in mdb.MD:
        md=mdb.MD[i]['medoid']
        md.plot("%smd_%d"%(outdir,i))
        md.save_mat("%smd_%d"%(outdir,i))

def plotCG(mdb,outdir):
    mdb.plot_CG("%sout_CG"%outdir)
#--------------------------------#


#--------------------------------#
def _set_env(cdir, animation):
    mscale=0
    while(True):
        dir_i='%s_Level%d'%(cdir,mscale)
        if(not tl.os.path.isdir(dir_i)): break
        mscale+=1
    global MSCALE_H; MSCALE_H=mscale
    global ANIMATION; ANIMATION=animation
    tl.msg('om_viz (set_env): animation=%d, mscale=%d'%(animation, mscale))
#--------------------------------#

#--------------------------------#
def _run_single(cdir, animation=ANIMATION):
    outdir="%s_viz/"%(cdir) 
    tl.mkdir("%s"%outdir)
    #--------------------------------#
    # (1) load modelDBs, Snaps
    mdb=tl.load_obj('%sMDB.obj'%(cdir))
    Snaps=tl.load_mat('%sSnaps.mat'%(cdir))
    CG=tl.load_obj('%sCGs.obj'%(cdir))
    # (2) save results 
    _save_all_L(Snaps, mdb, CG, outdir)
    #--------------------------------#

#--------------------------------#
def _run_mscale(cdir, animation=ANIMATION):
    outdir="%s_viz/"%(cdir) 
    tl.mkdir("%s"%outdir)
    #--------------------------------#
    # (1) load modelDBs, Snaps
    mdbHs={}; SnapsHs={}; CGHs={}; outdirHs={}
    for i in range(0,MSCALE_H):
        outdirHs[i]="%s_Level%d/"%(outdir,i)
        tl.mkdir("%s"%outdirHs[i])
        mdbHs[i]=tl.load_obj('%s_Level%d/MDB.obj'%(cdir,i))
        SnapsHs[i]=tl.load_mat('%s_Level%d/Snaps.mat'%(cdir,i))
        CGHs[i]=tl.load_obj('%s_Level%d/CGs.obj'%(cdir,i))
    #--------------------------------#
    # (2) save multi-scale (LOCAL) 
    arg_list=[]
    for i in range(0,MSCALE_H): arg_list.append([SnapsHs[i], mdbHs[i], CGHs[i], outdirHs[i]])
    if(MULTI): 
        _multi_save_all_L(arg_list)
    else: 
        for i in range(0,MSCALE_H): _save_all_L(SnapsHs[i], mdbHs[i], CGHs[i], outdirHs[i])
    #--------------------------------#
    # (3) save Xorg-cast (GLOBAL)
    Xorg=tl.loadsq('%sXorg.txt'%cdir).T # load Xorg
    Snaps=_save_all_G(Xorg, outdir, SnapsHs, mdbHs, outdir)
    #--------------------------------#
#--------------------------------#
def _multi_save_all_L(arg_list):
    n_proc=len(arg_list)
    pool = tl.multiprocessing.Pool(processes=n_proc)
    return pool.map(_wrapper_save_all_L, arg_list)
def _wrapper_save_all_L(arg_list):
    return _save_all_L(*arg_list)
#--------------------------------#




#--------------------------------#
# save all figs (individual-level)
def _save_all_L(Snaps, mdb, CGs, outdir):
    saveResults_txt(Snaps,mdb,outdir)
    plotCG(mdb,outdir)
    plotResultsEF(Snaps,mdb,outdir)
    if(FULLSAVE): plotResultsF(Snaps,mdb,outdir)
    if(FULLSAVE): plotResultsE(Snaps,mdb,outdir)
    if(ANIMATION): _plotCG_SS(Snaps, mdb, CGs, outdir) 
    plotMD(mdb,outdir)
    if(ANIMATION): _plotSS_L(Snaps, mdb, outdir)
    #--------------------------------#
    _save_html_L(Snaps, mdb, outdir)
    #--------------------------------#

#--------------------------------#
# save full results 
def _save_all_G(Xorg, cdir, SnapsHs, mdbHs, outdir):
    # compute Snaps[G] 
    lstep=SnapsHs[0]['lstep']
    multiscale_wd=1
    Snaps=orbitmap.init_Snaps(Xorg, lstep, multiscale_wd)
    Snaps['G']=tl.YES
    Snaps['Ve_full']=SnapsHs[0]['Ve_full']
    Snaps['Vf_full']=SnapsHs[0]['Vf_full']
    for i in range(1,MSCALE_H):
        Snaps['Ve_full']+=SnapsHs[i]['Ve_full']
        Snaps['Vf_full']+=SnapsHs[i]['Vf_full']
    Snaps['T_full']=np.nan*np.zeros((MSCALE_H,Snaps['n']))
    for i in range(0,MSCALE_H):
        Snaps['T_full'][i]=SnapsHs[i]['T_full']
    # save all
    _save_html_G(cdir, mdbHs)
    Snaps=orbitmap.compute_Snaps_Errs(Snaps)  
    tl.save_mat(Snaps, "%sSnaps"%(outdir))
    saveResults_txt(Snaps, None, outdir)
    # plot overview (full)
    _plotSS_t_G(Snaps['n'], Snaps, outdir, SnapsHs)    
    # (4) plot snaps (optional)
    if(ANIMATION): _plotSS_G(Snaps, outdir, SnapsHs)    
    return Snaps





#--------------------------------#
def _save_html_G(outdir, mdbHs):
    if(ANIMATION): 
        fp=tl.open_txt("%sresult_G_a.html"%(outdir))
    else: 
        fp=tl.open_txt("%sresult_G.html"%(outdir))
    tl.write_txt(fp,"<html>\n")
    #
    tl.write_txt(fp,"<h1> OrbitMap results</h1>\n")
    if(not ANIMATION): tl.write_txt(fp,"(mode: no animation)<br>\n")
    #
    tl.write_txt(fp,"<h2> Overview </h1>\n")
    tl.write_txt(fp,"<h3> Fitting result vs. original data stream </h3><br>\n")
    tl.write_txt(fp,"<img src=\"result_G.png\", height=400>\n")
    if(ANIMATION):
        tl.write_txt(fp,"<h2> Animation</h1>\n")
        tl.write_txt(fp,"<img src=\"SS_Vf_G.gif\", height=400>\n")
    tl.write_txt(fp,"<embed src=\"Snaps_out.txt\", height=400><br>\n")
    #
    tl.write_txt(fp,"<h2> Multi-scale fitting & forecasting</h1>\n") 
    tl.write_txt(fp,"<h3> Fitting result vs. original data stream, Dynamic space transitions </h3><br>\n")
    tl.write_txt(fp,"<table>\n")
    for i in range(0,MSCALE_H):
        if(mdbHs[i].get_c()>=MINC and mdbHs[i].CC==tl.YES): 
            tl.write_txt(fp,"<tr>\n")
            tl.write_txt(fp,"<th>Level_%d</th>\n"%(i))
            if(ANIMATION): 
                tl.write_txt(fp,"<th><img src=\"_Level%d/SS_Vf.gif\", height=300></th>\n"%(i))
                tl.write_txt(fp,"<th><img src=\"_Level%d/SS_CG.gif\", height=300></th>\n"%(i))
            else:
                if(not FULLSAVE): tl.write_txt(fp,"<th><img src=\"_Level%d/out_Vef.png\", height=300></th>\n"%(i))
                if(FULLSAVE): tl.write_txt(fp,"<th><img src=\"_Level%d/out_Ve.png\", height=300></th>\n"%(i))
                if(FULLSAVE): tl.write_txt(fp,"<th><img src=\"_Level%d/out_Vf.png\", height=300></th>\n"%(i))
                tl.write_txt(fp,"<th><img src=\"_Level%d/out_CG.png\", height=300></th>\n"%(i))
            tl.write_txt(fp,"<th><embed src=\"_Level%d/Snaps_out.txt\", height=200></th>\n"%(i))
            tl.write_txt(fp,"</tr>\n")
            tl.write_txt(fp,"\n")
    tl.write_txt(fp,"</table>\n")
    #
    tl.write_txt(fp,"<br>")
    tl.write_txt(fp,"</html>")
    tl.write_txt(fp,"")
    tl.close_txt(fp)
#--------------------------------#

#--------------------------------#
def _save_html_L(Snaps, mdb, outdir):
    if(ANIMATION): 
        fp=tl.open_txt("%sresult_L_a.html"%(outdir))
    else:
        fp=tl.open_txt("%sresult_L.html"%(outdir))
    tl.write_txt(fp,"<html>\n")
    tl.write_txt(fp,"<h1> OrbitMap result </h1><br>\n")
    if(not ANIMATION): tl.write_txt(fp,"(mode: no animation)<br>\n")
    #
    tl.write_txt(fp,"<embed src=\"Snaps_out.txt\"><br>\n")
    #
    tl.write_txt(fp,"<h3> Fitting result vs. original data stream, Dynamic space transitions </h3><br>\n")
    if(not FULLSAVE): tl.write_txt(fp,"<img src=\"out_Vef.png\", height=400>\n")
    if(FULLSAVE): tl.write_txt(fp,"<img src=\"out_Ve.png\", height=400>\n")
    if(FULLSAVE): tl.write_txt(fp,"<img src=\"out_Vf.png\", height=400>\n")
    tl.write_txt(fp,"<img  src=\"out_CG.png\", height=400>\n")
    tl.write_txt(fp,"<br>\n")
    tl.write_txt(fp,"<br>\n")
    if(ANIMATION): 
        tl.write_txt(fp,"<img src=\"SS_Vf.gif\", height=400>\n")
        tl.write_txt(fp,"<img src=\"SS_CG.gif\", height=400>\n")
    tl.write_txt(fp,"<br>")
    tl.write_txt(fp,"<br>")
    # 
    tl.write_txt(fp,"<h3> Typical regimes </h3><br>\n")
    for i in mdb.MD:
        tl.write_txt(fp,"<embed src=\"md_%d_plot.pdf\", height=300, width=200>\n"%(i))
    #
    tl.write_txt(fp,"<br>")
    tl.write_txt(fp,"</html>")
    tl.write_txt(fp,"")
    tl.close_txt(fp)
#--------------------------------#



def _plotCG_SS(Snaps, mdb, CGs, outdir):
    CGraph_org=mdb.create_CGraph_rcds()
    figfn_CG="%sSS_CG"%(outdir)
    #--------------------------------#
    nticks=len(Snaps['Re'])
    lstep=Snaps['lstep']
    pstep=Snaps['pstep']
    if(mdb.CC is tl.NO): nticks=1 # if, no Orbit-Chain at this level, just ignore
    for idx in range(0,nticks):
        #------------------------------------------------------#
        tb_st=Snaps['Te'][idx,0]; tb_ed=Snaps['Te'][idx,1]
        tp_st=Snaps['Te'][idx,2]; tp_ed=Snaps['Te'][idx,3]
        tc_st=Snaps['Te'][idx,4]; tc_ed=Snaps['Te'][idx,5]
        tf_st=Snaps['Te'][idx,6]; tf_ed=Snaps['Te'][idx,7]
        #if(DBG): tl.msg([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed, tf_st, tf_ed])
        #SSResults=tl.YES # if, want to plot snapshot-fit (default: no)
        SSResults=tl.NO   # if, plot full-fit result, i.e., Re_full, (default: yes) 
        if(SSResults):
            tb_ed-=tb_st;
            tp_st-=tb_st; tp_ed-=tb_st; 
            tc_st-=tb_st; tc_ed-=tb_st; 
            tf_st-=tb_st; tf_ed-=tb_st;
            tb_st=0; 
            if(DBG): tl.msg([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed, tf_st, tf_ed])
            Re_t=Snaps['Re'][idx] 
        else:
            Re_t=Snaps['Re_full']
        #------------------------------------------------------#
        #CGraph=tl.dcopy(CGraph_org) # if, static
        CGraph=CGs[idx] # if, snapshots
        #------------------------------------------------------#
        Re_t[np.isnan(Re_t)]=-1
        Re_t=Re_t.astype(int)
        # Vb
        nset=tl.collections.Counter(Re_t[tb_st:tb_ed])
        if(SSResults):
            del(nset[-1])
            for nid in nset: _updateCG_node(CGraph, int(nid), 'powderblue') 
            # Vp
            nset=tl.collections.Counter(Re_t[tp_st:tp_ed])
            del(nset[-1])
            for nid in nset: _updateCG_node(CGraph, int(nid), 'steelblue')
            # Vf
            nset=tl.collections.Counter(Re_t[tf_st:tf_ed])
            del(nset[-1])
            for nid in nset: _updateCG_node(CGraph, int(nid), 'lightpink')
            # Vc
            nset=tl.collections.Counter(Re_t[tc_st:tc_ed])
            del(nset[-1])
            for nid in nset: _updateCG_node(CGraph, int(nid), 'tomato')
        else:
            # Vc
            (Rc,Rp,Rb)=_find_two_steps_prev(Re_t, tc_ed) 
            # Vb
            #_updateCG_node(CGraph, Rb, 'powderblue')
            # Vp
            _updateCG_node(CGraph, Rp, 'powderblue') #'orange')
            # Vc
            _updateCG_node(CGraph, Rc, 'tomato')
            #_updateCG_node(CGraph, Re_t[tc_ed], 'tomato')
        #------------------------------------------------------#
        gv.write_dot(CGraph,"%s_tmp"%(figfn_CG))
        if(idx==0):
            tl.os.system("pdfunite %s_tmp.pdf %s.pdf"%(figfn_CG,figfn_CG))
        else:
            tl.os.system("cp %s.pdf %s_cp.pdf"%(figfn_CG,figfn_CG))
            tl.os.system("pdfunite %s_cp.pdf %s_tmp.pdf %s.pdf"%(figfn_CG,figfn_CG,figfn_CG))
    #--------------------------------#
    tl.os.system("convert  -loop 1 -dispose previous -resize 500 -density 200 %s.pdf %s.gif"%(figfn_CG,figfn_CG))
    #--------------------------------#

# create segID list
def _find_two_steps_prev(Re_full,tc):
    n=len(Re_full)
    Rc=Re_full[tc]; Rp=-1; Rb=-1;
    for t in range(0,tc):
        r_c=Re_full[tc-t]
        if(Rp==-1 and r_c != Rc): Rp=r_c
        if(Rp!=-1 and Rb==-1 and r_c!=Rp): Rb=r_c
    return (Rc, Rp, Rb)

# update node style (color)
def _updateCG_node(CGraph, nid, col):
    for node in CGraph['MD']:
        if(node['id']==nid): node['col']=col; break
    return CGraph




#--------------------------------#
def _plotSS_G(Snaps, outdir, SnapsHs):
    figfn_SS="%sSS_Vf_G"%(outdir)
    pp_SS=tl.pdfopen("%s.pdf"%(figfn_SS))
    for idx in range(0,len(SnapsHs[0]['Xe'])):
        tl.plt.clf()
        tc=SnapsHs[0]['Te'][idx,5]
        _plotSS_t_G(tc, Snaps, '', SnapsHs)
        pp_SS.savefig(tl.plt.gcf())
    pp_SS.close()
    tl.os.system("convert  -loop 1 -dispose previous %s.pdf %s.gif"%(figfn_SS,figfn_SS))
#--------------------------------#
def _plotSS_t_G(idx, Snaps, outdir, SnapsHs):
    tc=idx
    tf_st=tc+Snaps['lstep']
    tf_ed=tf_st+Snaps['pstep']
    n=Snaps['n']
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten());
    if(DBG): tl.msg([tc, tf_st, tf_ed])
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg[0:idx,:]) #, 'black')
    tl.plt.plot(range(idx,n), Xorg[idx:n,:], 'lightgrey')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    _plot_lines([tc], mn, mx, 0.5, 'b')
    _plot_lines([tf_st, tf_ed], mn, mx, 0.5, 'r')
    tl.plt.subplot(512)
    tl.plt.plot(Xorg[0:idx,:], 'lightgrey')
    tl.resetCol()
    tl.plt.plot(Snaps['Ve_full'][0:idx]) #, 'royalblue')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Estimation')
    tl.plt.xticks([])
    _plot_lines([tc], mn, mx, 0.5, 'b')
    _plot_lines([tf_st, tf_ed], mn, mx, 0.5, 'r')
    tl.plt.subplot(513)
    tl.plt.plot(Xorg[0:tf_ed,:], 'lightgrey')
    tl.resetCol()
    if(SMTH): tl.plt.plot(tl.smoothWMAo(Snaps['Vf_full'][0:tf_ed], int(Snaps['pstep'])))
    else: tl.plt.plot(Snaps['Vf_full'][0:tf_ed]) 
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Forecast')
    tl.plt.xticks([])
    _plot_lines([tc], mn, mx, 0.5, 'b')
    _plot_lines([tf_st, tf_ed], mn, mx, 0.5, 'r')
    tl.plt.subplot(514)
    tl.plt.plot(Snaps['Ef_full'][0:idx], 'yellowgreen')
    tl.plt.plot(Snaps['Es_full'][0:idx], 'lightgrey')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('RMSE')
    tl.plt.xticks([])
    emx=np.nanmax(Snaps['Es_full'].flatten())
    _plot_lines([tc], 0, emx, 0.5, 'b')
    _plot_lines([tf_st, tf_ed], 0, emx, 0.5, 'r')
    tl.plt.ylim([0,emx])
    tl.plt.subplot(515)
    tmx=-tl.INF; tmn=tl.INF
    for i in range(0,MSCALE_H):
        tl.plt.semilogy(SnapsHs[i]['T_full'][0:idx], '.', label='h=%d'%(i))
        mn=np.nanmin(SnapsHs[i]['T_full']); mx=np.nanmax(SnapsHs[i]['T_full']);
        if(tmx<mx): tmx=mx
        if(tmn>mn): tmn=mn
    tl.plt.xlim([0,len(Xorg)])
    _plot_lines([tc], tmn, tmx, 0.5, 'b')
    _plot_lines([tf_st, tf_ed], tmn, tmx, 0.5, 'r')
    tl.plt.ylim([tmn,tmx])
    tl.plt.ylabel('Speed')
    if(outdir!=''):
        tl.savefig("%sresult_G"%(outdir),'pdf')
        tl.savefig("%sresult_G"%(outdir),'png')



#--------------------------------#
def _plotSS_L(Snaps, mdb, outdir):
    figfn_SS="%sSS_Vf"%(outdir)
    pp_SS=tl.pdfopen("%s.pdf"%(figfn_SS))
    for idx in range(0,len(Snaps['Xe'])):
        tl.plt.clf()
        _plotSS_t_L(idx, Snaps, mdb)
        pp_SS.savefig(tl.plt.gcf())
    pp_SS.close()
    tl.os.system("convert  -loop 1 -dispose previous %s.pdf %s.gif"%(figfn_SS,figfn_SS))
#--------------------------------#
def _plotSS_t_L(idx, Snaps, mdb):
    #------------------------------------------------------#
    tb_st=Snaps['Te'][idx,0]; tb_ed=Snaps['Te'][idx,1]
    tp_st=Snaps['Te'][idx,2]; tp_ed=Snaps['Te'][idx,3]
    tc_st=Snaps['Te'][idx,4]; tc_ed=Snaps['Te'][idx,5]
    tf_st=Snaps['Te'][idx,6]; tf_ed=Snaps['Te'][idx,7]
    #------------------------------------------------------#
    if(DBG): tl.msg([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed, tf_st, tf_ed])
    Xe_t=Snaps['Xe'][idx]
    Ve_t=Snaps['Ve'][idx]
    Re_t=Snaps['Re'][idx]
    Xorg=Snaps['Xorg']
    nduration=Snaps['n']
    lstep=Snaps['lstep']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten());
    #-----------------------------------#
    tl.plt.clf()
    tl.plt.subplot(611)
    tl.plt.plot(Xorg) #, 'black')
    _plot_lines([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed], mn, mx, 0.5, 'b')    
    _plot_lines([tf_st, tf_ed], mn, mx, 0.5, 'r')    
    tl.plt.xlim([0,nduration])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(612)
    tl.plt.plot(Xorg, 'lightgrey', color=[0.8,0.8,0.8])
    tl.resetCol()
    if(SMTH): tl.plt.plot(range(0,tf_ed), tl.smoothWMAo(Snaps['Vf_full'][0:tf_ed], int(Snaps['pstep'])))
    else: tl.plt.plot(range(0,tf_ed), Snaps['Vf_full'][0:tf_ed]) 
    _plot_lines([tf_st, tf_ed], mn, mx, 0.5, 'r')    
    tl.plt.xlim([0,nduration])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Forecast')
    tl.plt.xticks([])
    tl.plt.subplot(613)
    _plot_lines([tb_st, tb_ed, tp_st, tp_ed, tc_st, tc_ed], -1, 100, 0.5, 'b')    
    tl.plt.imshow(_createImap(mdb, Snaps['Re_full'][0:tc_ed]), cmap='gray', interpolation='nearest', aspect='auto') 
    tl.plt.xlim([0,nduration]) #; tl.plt.yticks([])
    tl.plt.yticks(range(0,mdb.get_c()), range(1,mdb.get_c()+1))
    tl.plt.xticks([])
    tl.plt.ylabel('R-ID')
    #tl.plt.subplot(12,1,6)
    #_plot_lines([tf_st, tf_ed], -1, 100, 0.5, 'r')    
    #tl.plt.imshow(_createImap(mdb, Snaps['Rf_full'][0:tf_ed]), cmap='gray', interpolation='nearest', aspect='auto') 
    #tl.plt.xlim([0,nduration]) #; tl.plt.yticks([])
    #tl.plt.yticks(range(0,mdb.get_c()), range(1,mdb.get_c()+1))
    #-----------------------------------#
    ss_st=int(max(0,tc_ed-lstep*4)); ss_ed=int(min(tf_ed+lstep/2, nduration))
    #ss_st=int(max(0,tb_st-lstep*0)); ss_ed=int(min(tf_ed+lstep, nduration))
    #tl.plt.subplot(223)
    if(PLOT_R):  tl.plt.subplot(234)
    else: tl.plt.subplot(223)
    #tl.plt.plot(range(ss_st,ss_ed), Snaps['Xorg'][ss_st:ss_ed]) #, 'black') #,:])
    tl.plt.plot(range(ss_st,ss_ed), Snaps['Xorg'][ss_st:ss_ed], 'lightgrey')
    tl.resetCol()
    tl.plt.plot(range(ss_st,tc_ed), Snaps['Xorg'][ss_st:tc_ed]) #, 'black') #,:])
    _plot_lines([tb_st, tb_ed, tp_st, tp_ed, tc_st], mn, mx, 0.5, 'b')    
    _plot_lines([tc_ed, tf_st, tf_ed], mn, mx, 0.5, 'r')    
    tl.plt.xlim(ss_st, ss_ed); tl.plt.ylim([mn, mx])
    tl.plt.title('Org tc:%d, Vc [%d:%d]'%(tc_ed, tc_st,tc_ed))
    if(PLOT_R): tl.plt.subplot(235)
    else: tl.plt.subplot(224)
    tl.plt.plot(range(ss_st,ss_ed), Snaps['Xorg'][ss_st:ss_ed], 'lightgrey')
    tl.resetCol()
    tl.plt.plot(range(tb_st,tf_ed), Ve_t) #, 'tomato') 
    _plot_lines([tb_st, tb_ed, tp_st, tp_ed, tc_st], mn, mx, 0.5, 'b')    
    _plot_lines([tc_ed, tf_st, tf_ed], mn, mx, 0.5, 'r')    
    tl.plt.xlim(ss_st, ss_ed); tl.plt.ylim([mn, mx])
    tl.plt.title('Cast Vf [%d:%d]'%(tf_st, tf_ed))
    if(PLOT_R):
        tl.plt.subplot(236)
        tl.plt.plot(range(tb_st,tf_ed), Re_t)
        tl.plt.xlim(ss_st, ss_ed); 
    #-----------------------------------#
#--------------------------------#

#--------------------------------#
def _createImap(mdb, R):
    n=len(R) # sequence length
    if(mdb.CC is tl.NO): return np.zeros((n,1)) # if, no mdb, then return zero 
    c=mdb.get_c() # of cells 
    Rm=np.zeros((n,c)) # assigned ID list
    cnt=0
    for i in mdb.MD:
        for t in range(0,n):
            if(R[t]==i):
                Rm[t,cnt]=1
        cnt+=1
    Rm=(-1)*Rm[:,0:cnt].T
    return Rm
    
#--------------------------------#
def _plot_lines(Lines, mn, mx, wd, c):
    for i in Lines:
        tl.plt.plot([i,i],[mn,mx], '-', lw=wd, color=c)


#--------------------------------#
def _plotResultsEF(Snaps, mdb, outdir):
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten())
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg)
    tl.resetCol()
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(512)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    tl.plt.plot(Snaps['Ve_full']) 
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Estimation')
    tl.plt.xticks([])
    tl.plt.subplot(513)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    if(SMTH): tl.plt.plot(tl.smoothWMAo(Snaps['Vf_full'],int(Snaps['pstep'])))
    else: tl.plt.plot(Snaps['Vf_full']) 
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Forecast')
    tl.plt.xticks([])
    tl.plt.subplot(514)
    tl.plt.imshow(_createImap(mdb, Snaps['Re_full']), cmap='gray', interpolation='nearest', aspect='auto')
    tl.plt.xlim([0,len(Xorg)]) #; tl.plt.yticks([])
    tl.plt.yticks(range(0,mdb.get_c()), range(1,mdb.get_c()+1))
    tl.plt.ylabel('R-ID (est)')
    tl.plt.xticks([])
    tl.plt.subplot(515)
    tl.plt.semilogy(Snaps['T_full'], '.', color='yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('Speed')
    #tl.savefig("%sout_Vef"%(outdir),'pdf')
    tl.savefig("%sout_Vef"%(outdir),'png')
    tl.plt.close()


#--------------------------------#
def _plotResultsF(Snaps, mdb, outdir):
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten())
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg,) # 'black')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(512)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    if(SMTH): tl.plt.plot(tl.smoothWMAo(Snaps['Vf_full'],int(Snaps['pstep'])))
    else: tl.plt.plot(Snaps['Vf_full']) 
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Forecast')
    tl.plt.xticks([])
    tl.plt.subplot(513)
    tl.plt.plot(Snaps['Es_full'], 'lightgrey')
    tl.plt.plot(Snaps['Ef_full'], 'yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('RMSE (cast)')
    tl.plt.xticks([])
    emx=np.nanmax(Snaps['Es_full'].flatten())
    tl.plt.ylim([0,emx])
    tl.plt.subplot(514)
    #tl.plt.plot(Snaps['Rf_full'])
    tl.plt.imshow(_createImap(mdb, Snaps['Rf_full']), cmap='gray', interpolation='nearest', aspect='auto') 
    tl.plt.xlim([0,len(Xorg)]) #; tl.plt.yticks([])
    tl.plt.yticks(range(0,mdb.get_c()), range(1,mdb.get_c()+1))
    tl.plt.ylabel('R-ID (cast)')
    tl.plt.xticks([])
    tl.plt.subplot(515)
    tl.plt.semilogy(Snaps['T_full'], '.', color='yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('Speed')
    #tl.savefig("%sout_Vf"%(outdir),'pdf')
    tl.savefig("%sout_Vf"%(outdir),'png')
    tl.plt.close()

#--------------------------------#
def _plotResultsE(Snaps, mdb, outdir):
    Xorg=Snaps['Xorg']
    mn=np.nanmin(Xorg.flatten()); mx=np.nanmax(Xorg.flatten())
    tl.plt.clf()
    tl.plt.subplot(511)
    tl.plt.plot(Xorg) #, 'black')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Original')
    tl.plt.xticks([])
    tl.plt.subplot(512)
    tl.plt.plot(Xorg, 'lightgrey')
    tl.resetCol()
    tl.plt.plot(Snaps['Ve_full']) #, 'royalblue')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylim([mn, mx])
    tl.plt.ylabel('Est')
    tl.plt.xticks([])
    tl.plt.subplot(513)
    tl.plt.plot(Snaps['Ee_full'], 'yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    emx=np.nanmax(Snaps['Es_full'].flatten())
    tl.plt.ylim([0,emx])
    tl.plt.ylabel('RMSE (est)')
    tl.plt.xticks([])
    tl.plt.subplot(514)
    #tl.plt.plot(Snaps['Re_full'])
    tl.plt.imshow(_createImap(mdb, Snaps['Re_full']), cmap='gray', interpolation='nearest', aspect='auto')
    tl.plt.xlim([0,len(Xorg)]) #; tl.plt.yticks([])
    tl.plt.yticks(range(0,mdb.get_c()), range(1,mdb.get_c()+1))
    tl.plt.ylabel('R-ID (est)')
    tl.plt.xticks([])
    tl.plt.subplot(515)
    tl.plt.semilogy(Snaps['T_full'], '.', color='yellowgreen')
    tl.plt.xlim([0,len(Xorg)])
    tl.plt.ylabel('Speed')
    #tl.savefig("%sout_Ve"%(outdir),'pdf')
    tl.savefig("%sout_Ve"%(outdir),'png')
    tl.plt.close()
#--------------------------------#
 

#--------------------------------#
def _saveResults_txt(Snaps, mdb, outdir):
    fp=tl.open_txt("%sSnaps_out.txt"%(outdir))
    tl.write_txt(fp,"fn: %s\n"%(outdir))
    tl.write_txt(fp,"(n,d): (%d,%d)\n"%(Snaps['n'], Snaps['d']))
    tl.write_txt(fp,"(lstep,pstep,wd_level): (%d, %d, %d)\n"%(Snaps['lstep'], Snaps['pstep'], Snaps['wd_level']))
    if(mdb==None):
        tl.write_txt(fp,"MDB (None)\n")
    else:
        tl.write_txt(fp,"MDB (c): %d (CC:%d)\n"%(mdb.get_c(), mdb.CC))
        tl.write_txt(fp,"MDB (RE/RF/MS): %f/%f/%f\n"%(mdb.rho_RE, mdb.rho_RF, mdb.rho_MS))
    tl.write_txt(fp,"errE: %f\n"%(Snaps['errE']))
    tl.write_txt(fp,"errS: %f\n"%(Snaps['errS']))
    tl.write_txt(fp,"errF: %f\n"%(Snaps['errF']))
    tl.write_txt(fp,"errF (half): %f\n"%(Snaps['errF_half']))
    if('G' in Snaps):
        tl.write_txt(fp,"time: %f (max)\n"%(tl.mynanmean(np.amax(Snaps['T_full'],axis=0)))) 
        tl.write_txt(fp,"time: %f (mean)\n"%(tl.mynanmean(Snaps['T_full']))) 
    else: 
        tl.write_txt(fp,"time: %f \n"%(tl.mynanmean(Snaps['T_full']))) 
    tl.close_txt(fp)
#--------------------------------#




#---------------#
#     main      #
#---------------#
if __name__ == "__main__":
    tl.msg("OrbitMap (viz)")
   



